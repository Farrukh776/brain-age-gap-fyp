"""Final check that every preprocessed scan the models will read is ready for modelling.

Run from the repo root:  python -m src.preprocessing.verify_processed --workers 6

Read-only; prints counts and flagged image IDs only (image IDs, never subject IDs).
Re-run it whenever the processed scans are copied to another machine (e.g. the lab GPU).

Scans checked: all CN training scans (data/tables/cn_train_scans.csv) plus every CN val/test
and MCI scan in data/tables/scan_qc_final.csv with status not "excluded".

Per scan:   file opens and fully decompresses; shape, voxel size, orientation and origin match
            the template crop; float32; no NaN/inf; z-score inside the template brain (mean 0,
            SD 1); share of brain voxels stuck at the lowest / highest value (a large share
            means brain cut off by the field of view or padding); share of the top brain slab at
            the lowest value (top of head cut off); template correlation, compared with the
            value stored in qc_metrics.csv (shows the file is the final, reviewed version).
Across scans: correlation with the group mean image (misalignment or odd scans); correlation
            between a subject's later visits and its baseline (wrong person / misaligned visit);
            exact duplicate files and near-identical scans of different subjects.
            Same person enrolled under two subject IDs (same site, birth month and sex, and
            very similar images; decision 20).
FLAG lines are errors to fix. LOOK lines are scans to view by eye: most are strong atrophy,
weak contrast or a slightly different head size between scanners (all reviewed 2026-11-22).
Tables:     every scan has a file; subjects in one split and one cohort only; training scans
            belong to training subjects; age and MCI labels present.
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

from src.preprocessing.preprocess import load_template

TABLES = Path("data/tables")
POOL = 4  # mm; size of the blocks used for the cross-scan comparisons
_W: dict = {}


def reference(cfg: dict):
    template, mask, start = load_template(Path(cfg["paths"]["template_dir"]),
                                          cfg["output"]["shape"])
    crop = lambda img: sitk.RegionOfInterest(img, cfg["output"]["shape"], start)  # noqa: E731
    ref = crop(mask)
    brain = sitk.GetArrayFromImage(ref) > 0
    tmpl = sitk.GetArrayFromImage(crop(template))
    header = (ref.GetSize(), ref.GetSpacing(), ref.GetDirection(), ref.GetOrigin())
    return header, brain, tmpl


def pool(a: np.ndarray) -> np.ndarray:
    z, y, x = (s // POOL for s in a.shape)
    return a.reshape(z, POOL, y, POOL, x, POOL).mean(axis=(1, 3, 5))


def init(cfg: dict) -> None:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    header, brain, tmpl = reference(cfg)
    zs = np.where(brain.any(axis=(1, 2)))[0]          # array axis 0 = z (inferior..superior)
    top = np.zeros_like(brain)
    top[zs[-len(zs) // 10:]] = True                    # top 10% of the brain's height
    _W.update(cfg=cfg, header=header, brain=brain, tmpl=tmpl, top=brain & top,
              pooled_mask=pool(brain.astype(np.float32)) > 0.5)


def check_one(image_id: int) -> dict:
    w = _W
    path = Path(w["cfg"]["paths"]["out_dir"]) / f"{image_id}.nii.gz"
    row = {"image_id": image_id, "error": ""}
    try:
        raw = path.read_bytes()
        row["md5"] = hashlib.md5(raw).hexdigest()
        img = sitk.ReadImage(str(path))
        a = sitk.GetArrayFromImage(img)
    except Exception as e:  # noqa: BLE001
        row["error"] = f"read: {type(e).__name__}"
        return row
    size, spacing, direction, origin = w["header"]
    row["header_ok"] = (img.GetSize() == size
                        and np.allclose(img.GetSpacing(), spacing, atol=1e-4)
                        and np.allclose(img.GetDirection(), direction, atol=1e-4)
                        and np.allclose(img.GetOrigin(), origin, atol=1e-3))
    row["float32"] = a.dtype == np.float32
    row["finite"] = bool(np.isfinite(a).all())
    if not (row["header_ok"] and row["finite"]):
        return row
    b = a[w["brain"]]
    lo, hi = float(a.min()), float(a.max())
    row.update(brain_mean=float(b.mean()), brain_sd=float(b.std()), vmin=lo, vmax=hi,
               brain_at_min=float((b <= lo + 1e-6).mean()),
               brain_at_max=float((b >= hi - 1e-6).mean()),
               top_at_min=float((a[w["top"]] <= lo + 1e-6).mean()),
               template_corr=float(np.corrcoef(b, w["tmpl"][w["brain"]])[0, 1]))
    row["vec"] = pool(a)[w["pooled_mask"]].astype(np.float32)
    return row


def load_scans() -> pd.DataFrame:
    train = pd.read_csv(TABLES / "cn_train_scans.csv")
    train = train.assign(cohort="CN", split="train")
    qc = pd.read_csv(TABLES / "scan_qc_final.csv")
    qc = qc[(qc["status"] != "excluded") & ~((qc["cohort"] == "CN") & (qc["split"] == "train"))]
    qc = qc.assign(image_id=qc["image_id"].astype("int64"), is_baseline=True)
    cols = ["cohort", "split", "RID", "image_id", "is_baseline"]
    return pd.concat([train[cols + ["age"]], qc[cols]], ignore_index=True)


def check_tables(scans: pd.DataFrame, out_dir: Path) -> list:
    problems = []
    cn_s = pd.read_csv(TABLES / "cn_splits.csv").set_index("RID")["split"]
    mci_s = pd.read_csv(TABLES / "mci_splits.csv").set_index("RID")["split"]
    mci = pd.read_csv(TABLES / "mci_cohort.csv")
    cn = pd.read_csv(TABLES / "cn_cohort.csv")

    if not scans["image_id"].is_unique:
        problems.append("an image is listed twice")
    missing = [i for i in scans["image_id"] if not (out_dir / f"{i}.nii.gz").exists()]
    if missing:
        problems.append(f"{len(missing)} scans have no processed file")
    per_rid = scans.groupby("RID")[["cohort", "split"]].nunique()
    if (per_rid > 1).any().any():
        problems.append("a subject appears in more than one split or cohort")
    if set(cn_s.index) & set(mci_s.index):
        problems.append("a subject is in both CN and MCI split tables")
    for name, s, sp in (("CN", scans[scans.cohort == "CN"], cn_s),
                        ("MCI", scans[scans.cohort == "MCI"], mci_s)):
        if (s["RID"].map(sp) != s["split"]).any():
            problems.append(f"{name} scan split does not match {name.lower()}_splits.csv")
    cn_age = scans[(scans.cohort == "CN")]["image_id"].map(cn.set_index("image_id")["age"])
    if cn_age.isna().any():
        problems.append(f"{int(cn_age.isna().sum())} CN scans have no age")
    m = scans[scans.cohort == "MCI"]["RID"].map(mci.drop_duplicates("RID").set_index("RID")["label"])
    if m.isna().any():
        problems.append(f"{int(m.isna().sum())} MCI scans have no label")
    t = scans[(scans.cohort == "CN") & (scans.split == "train")]
    if (t.groupby("RID")["is_baseline"].sum() != 1).any():
        problems.append("a CN training subject does not have exactly one baseline scan")
    return problems


SAME_PERSON_R = 0.85


def same_person_candidates(res: pd.DataFrame) -> list:
    """Baseline image-ID pairs of different subjects with the same site, birth month and sex."""
    raw = Path(yaml.safe_load(open("configs/cohort.yaml"))["paths"]["raw_tables"])
    dem = pd.read_csv(next(raw.glob("PTDEMOG_*.csv")), low_memory=False,
                      usecols=["RID", "PTDOB", "PTGENDER"]).drop_duplicates("RID", keep="last")
    ptid = pd.concat([pd.read_csv(TABLES / f, usecols=["RID", "PTID"])
                      for f in ("cn_cohort.csv", "mci_cohort.csv")]).drop_duplicates("RID")
    base = res[res["is_baseline"]][["RID", "image_id"]].merge(dem, on="RID").merge(ptid, on="RID")
    base["site"] = base["PTID"].str[:3]
    pairs = []
    for _, g in base.dropna(subset=["PTDOB"]).groupby(["site", "PTDOB", "PTGENDER"]):
        ids = g["image_id"].tolist()
        pairs += [(ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))]
    return pairs


def robust_low(x: pd.Series, k: float = 5.0) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return x < med - k * mad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save", help="optional CSV path for per-scan results (keep it local)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/preprocessing.yaml"))
    out_dir = Path(cfg["paths"]["out_dir"])

    scans = load_scans()
    print(f"scans to check: {len(scans)} "
          f"({scans.groupby(['cohort', 'split']).size().to_dict()})")
    problems = check_tables(scans, out_dir)
    print("table checks:", "all passed" if not problems else problems)
    used = set(scans["image_id"])
    extra = [p for p in out_dir.glob("*.nii.gz") if int(p.name.split(".")[0]) not in used]
    print(f"processed files not used by any table (excluded or replaced scans): {len(extra)}")

    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(cfg,)) as ex:
        rows = list(ex.map(check_one, scans["image_id"], chunksize=8))
    vecs = {r["image_id"]: r.pop("vec") for r in rows if "vec" in r}
    res = scans.merge(pd.DataFrame(rows), on="image_id")

    flags: dict = {}

    def flag(name, mask):
        flags[name] = res.loc[mask.fillna(False).astype(bool), "image_id"].tolist()

    flag("read error", res["error"] != "")
    flag("header (shape/spacing/orientation/origin) differs", res["header_ok"] != True)  # noqa: E712
    flag("not float32", res["float32"] != True)  # noqa: E712
    flag("NaN or inf values", res["finite"] != True)  # noqa: E712
    flag("brain mean not 0 (|mean| > 0.01)", res["brain_mean"].abs() > 0.01)
    flag("brain SD not 1 (off by > 0.01)", (res["brain_sd"] - 1).abs() > 0.01)
    flag("more than 2% of brain at lowest value", res["brain_at_min"] > 0.02)
    flag("more than 2% of brain at highest value", res["brain_at_max"] > 0.02)
    # Atrophied brains reach about 8% (dark CSF over the cortex); a cut-off top is far higher.
    flag("more than 25% of top brain slab at lowest value", res["top_at_min"] > 0.25)
    flag("LOOK template correlation below 0.30", res["template_corr"] < 0.30)

    stored = pd.read_csv(Path(cfg["paths"]["qc_dir"]) / "qc_metrics.csv")
    stored = stored.drop_duplicates("image_id", keep="last").set_index("image_id")
    diff = (res["template_corr"] - res["image_id"].map(stored["template_correlation"])).abs()
    flag("file differs from final QC record (correlation)", diff > 0.005)

    # Cross-scan checks on 4 mm block averages inside the brain
    ids = list(vecs)
    X = np.stack([vecs[i] for i in ids])
    Xc = X - X.mean(axis=1, keepdims=True)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
    mean = Xc.mean(axis=0)
    gcorr = pd.Series(Xc @ (mean / np.linalg.norm(mean)), index=ids)
    res["group_corr"] = res["image_id"].map(gcorr)
    flag("LOOK unusually low correlation with group mean", robust_low(res["group_corr"]))

    sim = Xc @ Xc.T
    rid = res.set_index("image_id").loc[ids, "RID"].to_numpy()
    pos = {i: k for k, i in enumerate(ids)}
    base = res[res["is_baseline"]].set_index("RID")["image_id"]
    later = res[~res["is_baseline"]]
    wcorr = pd.Series({i: sim[pos[i], pos[base[r]]] for i, r in zip(later["image_id"], later["RID"])})
    res["visit_corr"] = res["image_id"].map(wcorr)
    flag("LOOK later visit unlike the subject's baseline", robust_low(res["visit_corr"].dropna())
         .reindex(res.index, fill_value=False))
    other = rid[:, None] != rid[None, :]
    iu = np.triu_indices(len(ids), 1)
    dup_md5 = res[res["md5"].duplicated(keep=False) & res["md5"].notna()]["image_id"].tolist()
    flags["identical files"] = dup_md5

    # Same person enrolled under two subject IDs: same site, birth month and sex, and baseline
    # images far more alike than any two different people (found 2026-11-22, decision 19).
    near = []
    for a, b in same_person_candidates(res):
        r = float(sim[pos[a], pos[b]])
        if r > SAME_PERSON_R:
            near.append((a, b))
    flags[f"same person under two subject IDs (r > {SAME_PERSON_R})"] = [f"{a}~{b}" for a, b in near]

    if args.save:
        res.drop(columns=["RID"]).to_csv(args.save, index=False)
        pd.Series(dict(enumerate(f"{a}~{b}~{sim[pos[a], pos[b]]:.4f}" for a, b in near)),
                  dtype=object).to_csv(Path(args.save).with_suffix(".pairs.csv"))

    # Report
    print("\nsummary over scans (min / median / max):")
    for c in ["brain_mean", "brain_sd", "vmin", "vmax", "brain_at_min", "brain_at_max",
              "top_at_min", "template_corr", "group_corr", "visit_corr"]:
        s = res[c].dropna()
        print(f"  {c:15s} {s.min():8.3f} {s.median():8.3f} {s.max():8.3f}")
    other_max = sim[iu][other[iu]].max()
    print(f"  highest correlation between scans of different subjects: {other_max:.3f}")
    print("\nchecks:")
    for name, lst in flags.items():
        look = name.startswith("LOOK ")
        tag = "OK  " if not lst else ("LOOK" if look else "FLAG")
        print(f"  {tag} {name.removeprefix('LOOK ')}: {len(lst)}"
              + (f"  -> {lst[:20]}" if lst else ""))


if __name__ == "__main__":
    main()
