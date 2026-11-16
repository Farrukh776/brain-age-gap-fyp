"""Build the brain-age (CN) and progression (MCI) cohort tables.

Run from the repo root:  python -m src.cohort.build_cohorts

Rules come from docs/cohort_definition.md and configs/cohort.yaml (decisions 7 to 12).
Writes data/tables/cn_cohort.csv and data/tables/mci_cohort.csv (local only, never committed)
and prints counts only. It does not create the train / validation / test splits.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG = Path("configs/cohort.yaml")
CN, MCI, DEMENTIA = 1, 2, 3
DAYS_PER_MONTH = 30.44
NOT_T1 = r"localizer|scout|survey|calibration|flair|t2"  # decision 15


def find_table(folder: Path, name: str) -> Path:
    """Return the newest file called <name>_<date>.csv in the folder."""
    matches = sorted(folder.glob(f"{name}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No {name}_*.csv in {folder}")
    return matches[-1]


def read(folder: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(find_table(folder, name), low_memory=False)


# ---------------------------------------------------------------- diagnoses
def load_diagnoses(folder: Path) -> pd.DataFrame:
    """One row per diagnosis visit, with a date and months since the subject's baseline."""
    dx = read(folder, "DXSUM")
    # Some rows have no exam date; take it from the visit registry where possible
    key = ["RID", "PHASE", "VISCODE"]
    reg = read(folder, "REGISTRY").dropna(subset=["EXAMDATE"]).drop_duplicates(key)
    dx = dx.merge(reg[key + ["EXAMDATE"]], on=key, how="left", suffixes=("", "_REG"))
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"].fillna(dx["EXAMDATE_REG"]), errors="coerce")
    dx = dx[dx["DIAGNOSIS"].isin([CN, MCI, DEMENTIA]) & dx["EXAMDATE"].notna()]
    dx = dx.sort_values(["RID", "EXAMDATE"]).reset_index(drop=True)

    first = dx.groupby("RID").first()
    dx["BL_DATE"] = dx["RID"].map(first["EXAMDATE"])
    dx["BL_DX"] = dx["RID"].map(first["DIAGNOSIS"])
    dx["BL_PHASE"] = dx["RID"].map(first["PHASE"])

    # Months since baseline: use the scheduled visit code (sc/bl = 0, m36 = 36), because
    # visits rarely fall on the exact day. Fall back to dates, rounded to 6 months.
    code = dx["VISCODE2"].astype(str).str.lower()
    nominal = pd.to_numeric(code.str.extract(r"^m(\d+)$")[0], errors="coerce")
    nominal = nominal.where(~code.isin(["sc", "bl"]), 0)
    by_date = ((dx["EXAMDATE"] - dx["BL_DATE"]).dt.days / DAYS_PER_MONTH / 6).round() * 6
    first_nominal = dx["RID"].map(nominal.groupby(dx["RID"]).first())
    dx["MONTHS"] = (nominal - first_nominal).fillna(by_date).clip(lower=0)
    return dx


# ---------------------------------------------------------------- scans
def load_sessions(folder: Path) -> pd.DataFrame:
    """One row per scan session (subject + date): the chosen T1 image and its field strength."""
    man = read(folder, "Clinical_T1w_Imaging_Cohort_Manifest")
    man["image_date"] = pd.to_datetime(man["image_date"], errors="coerce")
    # The T1 manifest holds a few non-T1 images (a localizer, a FLAIR); never choose them
    man = man[~man["image_description"].str.contains(NOT_T1, case=False, na=False)]
    desc = man["image_description"].str.lower()
    man["repeat"] = desc.str.contains("repeat")
    man["accelerated"] = desc.str.contains("accel|grappa|sense|cs |hs ")
    # One image per session: prefer the first (non-repeat), non-accelerated acquisition
    man = man.sort_values(["subject_id", "image_date", "repeat", "accelerated", "image_id"])
    ses = man.drop_duplicates(["subject_id", "image_date"]).copy()

    parts = []
    for name in ("MRIMETA", "MRI3META"):
        part = read(folder, name)
        # Parse dates per table: the two tables do not use the same date format
        part["EXAMDATE"] = pd.to_datetime(part["EXAMDATE"], errors="coerce")
        parts.append(part[["PTID", "EXAMDATE", "FIELD_STRENGTH", "MMCONDCT"]])
    meta = pd.concat(parts, ignore_index=True)
    # 0 = scan not conducted. ADNI1 screening scans carry -4 (not recorded) and are kept.
    meta = meta[(meta["MMCONDCT"] != 0) & meta["EXAMDATE"].notna()]

    # 1) Subjects only ever scanned at one field strength
    per_subject = meta.groupby("PTID")["FIELD_STRENGTH"].agg(lambda s: set(s))
    single = per_subject[per_subject.str.len() == 1].str.join("")
    ses["field_strength"] = ses["subject_id"].map(single)

    # 2) Other subjects: the scan-information rows closest in date (at most 7 days away),
    #    if they agree on one strength
    todo = ses[ses["field_strength"].isna()][["subject_id", "image_date"]]
    near = todo.merge(meta, left_on="subject_id", right_on="PTID")
    near["gap"] = (near["image_date"] - near["EXAMDATE"]).dt.days.abs()
    closest = near.groupby(["subject_id", "image_date"])["gap"].transform("min")
    near = near[(near["gap"] <= 7) & (near["gap"] == closest)]
    agreed = near.groupby(["subject_id", "image_date"])["FIELD_STRENGTH"].agg(lambda s: set(s))
    agreed = agreed[agreed.str.len() == 1].str.join("").rename("fs_meta").reset_index()
    ses = ses.merge(agreed, on=["subject_id", "image_date"], how="left")
    ses["field_strength"] = ses["field_strength"].fillna(ses["fs_meta"])

    # 3) Still unknown: the FreeSurfer table records field strength per image
    fsx = read(folder, "UCSFFSX7").drop_duplicates("IMAGEUID").set_index("IMAGEUID")
    man["fs_image"] = man["image_id"].map(fsx["FIELD_STRENGTH"])
    by_image = man.dropna(subset=["fs_image"]).groupby(["subject_id", "image_date"])["fs_image"]
    by_image = by_image.agg(lambda s: set(s))
    by_image = by_image[by_image.str.len() == 1].str.join("").rename("fs_img").reset_index()
    ses = ses.merge(by_image, on=["subject_id", "image_date"], how="left")
    ses["field_strength"] = ses["field_strength"].fillna(ses["fs_img"])

    print("== Scan sessions ==")
    print("sessions:", len(ses), "| field strength:",
          ses["field_strength"].value_counts(dropna=False).to_dict())
    cols = ["subject_id", "image_id", "series_id", "image_date", "image_description",
            "field_strength"]
    return ses[cols].rename(columns={"subject_id": "PTID", "image_date": "scan_date"})


# ---------------------------------------------------------------- demographics
def load_demographics(folder: Path) -> pd.DataFrame:
    """One row per subject: date of birth (mid-month), sex, years of education."""
    pt = read(folder, "PTDEMOG").sort_values(["RID", "VISDATE"])
    pt["dob"] = pd.to_datetime("15/" + pt["PTDOB"].astype(str), format="%d/%m/%Y", errors="coerce")
    pt["sex"] = pt["PTGENDER"].map({1: "M", 2: "F"})
    pt["education"] = pt["PTEDUCAT"].where(pt["PTEDUCAT"] >= 0)
    # A subject can have a row per phase; keep the first recorded value of each field
    return pt.groupby("RID")[["dob", "sex", "education"]].first()


def baseline_mmse(folder: Path, baseline: pd.Series, max_days: int) -> pd.Series:
    """MMSE total score closest to each subject's baseline date."""
    mm = read(folder, "MMSE")
    mm = mm[mm["MMSCORE"].between(0, 30)].copy()
    mm["VISDATE"] = pd.to_datetime(mm["VISDATE"], errors="coerce")
    mm["days"] = (mm["VISDATE"] - mm["RID"].map(baseline)).dt.days.abs()
    mm = mm[mm["days"] <= max_days].sort_values("days").drop_duplicates("RID")
    return mm.set_index("RID")["MMSCORE"]


# ---------------------------------------------------------------- cohorts
def attach_baseline_scan(subjects: pd.DataFrame, ses: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add the nearest allowed scan within the day limit; subjects without one are dropped."""
    wanted = ses[ses["field_strength"] == cfg["field_strength"]]
    m = subjects.merge(wanted, on="PTID")
    m["days_from_baseline"] = (m["scan_date"] - m["BL_DATE"]).dt.days
    m["gap"] = m["days_from_baseline"].abs()
    m = m[m["gap"] <= cfg["max_days_from_baseline"]].sort_values(["RID", "gap"])
    return m.drop_duplicates("RID").drop(columns="gap")


def add_person_columns(df: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    df = df.join(demo, on="RID")
    df["age"] = ((df["scan_date"] - df["dob"]).dt.days / 365.25).round(2)
    return df.drop(columns="dob")


def flow(title: str, steps: list) -> None:
    print(f"\n== {title} ==")
    for label, n in steps:
        print(f"{n:6d}  {label}")


def build_cn(dx, ses, demo, cfg) -> pd.DataFrame:
    rules, img = cfg["brain_age"], cfg["imaging"]
    subj = dx[dx["BL_DX"] == CN]
    steps = [("CN at baseline", subj["RID"].nunique())]
    subj = subj[subj["BL_PHASE"].isin(rules["phases"])]
    steps.append(("in allowed phases", subj["RID"].nunique()))
    if rules["exclude_later_converters"]:
        converters = set(subj[subj["DIAGNOSIS"] >= MCI]["RID"])
        subj = subj[~subj["RID"].isin(converters)]
        steps.append(("never diagnosed MCI or AD later", subj["RID"].nunique()))
    last_dx = subj.groupby("RID")["EXAMDATE"].max().rename("LAST_DX")
    base = subj.drop_duplicates("RID")[["RID", "PTID", "BL_PHASE", "BL_DATE"]]

    bl = attach_baseline_scan(base, ses, img)
    steps.append((f"{img['field_strength']} scan within {img['max_days_from_baseline']} days",
                  len(bl)))
    bl = add_person_columns(bl, demo)
    bl = bl[bl["age"].notna() & bl["sex"].notna()]
    steps.append(("known age and sex", len(bl)))
    flow("Brain-age cohort (subjects)", steps)

    # All scans of the kept subjects while they are followed as CN (for the training set)
    scans = base[base["RID"].isin(bl["RID"])].merge(
        ses[ses["field_strength"] == img["field_strength"]], on="PTID").join(last_dx, on="RID")
    margin = pd.Timedelta(days=img["max_days_from_baseline"])
    scans = scans[(scans["scan_date"] >= scans["BL_DATE"] - margin)
                  & (scans["scan_date"] <= scans["LAST_DX"] + margin)]
    scans["days_from_baseline"] = (scans["scan_date"] - scans["BL_DATE"]).dt.days
    scans = add_person_columns(scans, demo)
    scans["is_baseline"] = scans["image_id"].isin(bl["image_id"])
    print(f"{len(scans):6d}  scans in total ({int(scans['is_baseline'].sum())} baseline)")
    print("age: mean %.1f, min %.1f, max %.1f | sex: %s" % (
        bl["age"].mean(), bl["age"].min(), bl["age"].max(), bl["sex"].value_counts().to_dict()))
    print("subjects by phase:", bl["BL_PHASE"].value_counts().to_dict())
    cols = ["RID", "PTID", "BL_PHASE", "image_id", "series_id", "scan_date", "image_description",
            "field_strength", "days_from_baseline", "is_baseline", "age", "sex", "education"]
    return scans[cols].sort_values(["RID", "scan_date"])


def build_mci(dx, ses, demo, mmse, cfg) -> pd.DataFrame:
    rules, img = cfg["progression"], cfg["imaging"]
    window = rules["window_months"]
    subj = dx[dx["BL_DX"] == MCI]
    steps = [("MCI at baseline", subj["RID"].nunique())]
    subj = subj[subj["BL_PHASE"].isin(rules["phases"])]
    steps.append(("in allowed phases", subj["RID"].nunique()))

    in_window = subj[subj["MONTHS"] <= window]
    progressors = set(in_window[in_window["DIAGNOSIS"] == DEMENTIA]["RID"])
    reverters = set(subj[subj["DIAGNOSIS"] == CN]["RID"]) - progressors
    followup = subj.groupby("RID")["MONTHS"].max()
    if rules["exclude_reverters"]:
        subj = subj[~subj["RID"].isin(reverters)]
        steps.append(("not reverting to CN", subj["RID"].nunique()))
    short = set(followup[followup < window].index) - progressors
    subj = subj[~subj["RID"].isin(short)]
    steps.append((f"progressed, or followed for {window}+ months", subj["RID"].nunique()))

    base = subj.drop_duplicates("RID")[["RID", "PTID", "BL_PHASE", "BL_DATE"]].copy()
    base["label"] = base["RID"].isin(progressors).astype(int)
    base["followup_months"] = base["RID"].map(followup)
    out = attach_baseline_scan(base, ses, img)
    steps.append((f"{img['field_strength']} scan within {img['max_days_from_baseline']} days",
                  len(out)))
    out = add_person_columns(out, demo)
    out = out[out["age"].notna() & out["sex"].notna()]
    steps.append(("known age and sex", len(out)))
    flow("Progression cohort (subjects)", steps)

    out["mmse"] = out["RID"].map(mmse)
    print("labels: stable %d, progressor %d (%.0f%% progressors)" % (
        (out["label"] == 0).sum(), (out["label"] == 1).sum(), 100 * out["label"].mean()))
    print("by phase:", pd.crosstab(out["BL_PHASE"], out["label"]).to_dict())
    print("age by label:", out.groupby("label")["age"].mean().round(1).to_dict(),
          "| MMSE by label:", out.groupby("label")["mmse"].mean().round(1).to_dict(),
          "| MMSE missing:", int(out["mmse"].isna().sum()))
    cols = ["RID", "PTID", "BL_PHASE", "image_id", "series_id", "scan_date", "image_description",
            "field_strength", "days_from_baseline", "age", "sex", "education", "mmse",
            "followup_months", "label"]
    return out[cols].sort_values("RID")


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    folder, out_dir = Path(cfg["paths"]["raw_tables"]), Path(cfg["paths"]["out_dir"])
    dx = load_diagnoses(folder)
    ses = load_sessions(folder)
    demo = load_demographics(folder)
    baseline = dx.drop_duplicates("RID").set_index("RID")["BL_DATE"]
    mmse = baseline_mmse(folder, baseline, cfg["imaging"]["max_days_from_baseline"])

    cn = build_cn(dx, ses, demo, cfg)
    mci = build_mci(dx, ses, demo, mmse, cfg)

    overlap = set(cn["RID"]) & set(mci["RID"])
    assert not overlap, "A subject is in both cohorts"
    assert mci["RID"].is_unique and cn["image_id"].is_unique
    print("\nsubjects in both cohorts:", len(overlap))

    out_dir.mkdir(parents=True, exist_ok=True)
    cn.to_csv(out_dir / "cn_cohort.csv", index=False)
    mci.to_csv(out_dir / "mci_cohort.csv", index=False)
    print("written:", out_dir / "cn_cohort.csv", "and", out_dir / "mci_cohort.csv")


if __name__ == "__main__":
    main()
