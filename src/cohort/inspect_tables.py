"""Print a counts-only overview of the raw ADNI clinical tables.

Run from the repo root:  python -m src.cohort.inspect_tables

Prints table sizes, diagnosis codes and aggregate subject counts. It never prints
subject IDs or subject-level rows. The numbers are exploratory: "baseline" here means
each subject's first visit with a recorded diagnosis, which is a working definition
until docs/cohort_definition.md is final.
"""
from pathlib import Path

import pandas as pd
import yaml

CONFIG = Path("configs/cohort.yaml")
TABLES = ["DXSUM", "PTDEMOG", "MMSE", "REGISTRY"]
DX_NAMES = {1: "CN", 2: "MCI", 3: "AD/dementia"}


def find_table(folder: Path, name: str) -> Path:
    """Return the newest file called <name>_<date>.csv in the folder."""
    matches = sorted(folder.glob(f"{name}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No {name}_*.csv in {folder}")
    return matches[-1]


def overview(folder: Path) -> None:
    print("== Table sizes ==")
    for name in TABLES:
        df = pd.read_csv(find_table(folder, name), low_memory=False)
        print(f"{name}: {len(df)} rows, {df['RID'].nunique()} subjects, {df.shape[1]} columns")
        print("  rows per phase:", df["PHASE"].value_counts().to_dict())


def diagnosis_codes(folder: Path) -> None:
    dic = pd.read_csv(find_table(folder, "DATADIC"), low_memory=False)
    rows = dic[(dic["TBLNAME"] == "DXSUM") & (dic["FLDNAME"] == "DIAGNOSIS")]
    print("\n== DIAGNOSIS coding (from the data dictionary) ==")
    for code in rows["CODE"].dropna().unique():
        print(" ", code)


def baseline_counts(folder: Path, window_months: int) -> None:
    dx = pd.read_csv(find_table(folder, "DXSUM"), low_memory=False)
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    print("\n== Diagnosis rows ==")
    print("missing DIAGNOSIS:", int(dx["DIAGNOSIS"].isna().sum()),
          "| missing EXAMDATE:", int(dx["EXAMDATE"].isna().sum()))

    dx = dx.dropna(subset=["DIAGNOSIS", "EXAMDATE"]).sort_values(["RID", "EXAMDATE"])
    first = dx.groupby("RID").first()
    last = dx.groupby("RID").last()

    print("\n== Baseline diagnosis per subject, by phase of first visit ==")
    table = pd.crosstab(first["PHASE"], first["DIAGNOSIS"].map(DX_NAMES), margins=True)
    print(table.to_string())

    # Months from each subject's baseline to each later visit
    dx["EXAMDATE_BL"] = dx["RID"].map(first["EXAMDATE"])
    dx["DIAGNOSIS_BL"] = dx["RID"].map(first["DIAGNOSIS"])
    dx["MONTHS"] = (dx["EXAMDATE"] - dx["EXAMDATE_BL"]).dt.days / 30.44
    followup = dx.groupby("RID")["MONTHS"].max()

    # CN at baseline: who is ever diagnosed MCI or AD later?
    cn = dx[dx["DIAGNOSIS_BL"] == 1]
    cn_converts = cn[cn["DIAGNOSIS"] >= 2]["RID"].nunique()
    print("\n== CN at baseline ==")
    print("subjects:", cn["RID"].nunique(), "| later MCI or AD:", cn_converts,
          "| CN visits in total:", int((cn["DIAGNOSIS"] == 1).sum()))

    # MCI at baseline: outcome inside the follow-up window
    mci = dx[dx["DIAGNOSIS_BL"] == 2]
    in_window = mci[mci["MONTHS"] <= window_months]
    progressors = set(in_window[in_window["DIAGNOSIS"] == 3]["RID"])
    reverters = set(mci[mci["DIAGNOSIS"] == 1]["RID"]) - progressors
    rest = set(mci["RID"]) - progressors - reverters
    enough = {rid for rid in rest if followup[rid] >= window_months}
    print(f"\n== MCI at baseline, {window_months}-month window (exploratory) ==")
    print("subjects:", mci["RID"].nunique())
    print("progress to AD within window:", len(progressors))
    print("revert to CN at any visit (not progressors):", len(reverters))
    print("stable with follow-up >= window:", len(enough))
    print("follow-up too short, no conversion:", len(rest) - len(enough))

    groups = {
        "CN, stays CN": set(cn["RID"]) - set(cn[cn["DIAGNOSIS"] >= 2]["RID"]),
        "CN, later MCI/AD": set(cn[cn["DIAGNOSIS"] >= 2]["RID"]),
        "MCI progressor": progressors,
        "MCI stable": enough,
    }
    return first, groups


def scan_availability(folder: Path, first: pd.DataFrame, groups: dict, max_days: int = 90) -> None:
    """Count subjects with a T1 scan, using the Clinical T1w Imaging Cohort manifest."""
    man = pd.read_csv(find_table(folder, "Clinical_T1w_Imaging_Cohort_Manifest"))
    man["image_date"] = pd.to_datetime(man["image_date"], errors="coerce")
    print("\n== T1 image manifest ==")
    print(f"{len(man)} images, {man['subject_id'].nunique()} subjects, "
          f"{man.groupby(['subject_id', 'image_date']).ngroups} scan sessions")

    # The manifest identifies subjects by PTID; the clinical tables are keyed by RID
    base = first.reset_index()[["RID", "PTID", "PHASE", "EXAMDATE"]]
    man = man.merge(base, left_on="subject_id", right_on="PTID", how="inner")
    man["DAYS"] = (man["image_date"] - man["EXAMDATE"]).dt.days.abs()
    any_scan = set(man["RID"])
    near = man[man["DAYS"] <= max_days]
    bl_scan = set(near["RID"])
    phase = base.set_index("RID")["PHASE"]

    print(f"\n== Subjects with a T1 scan (baseline scan = within {max_days} days of baseline diagnosis) ==")
    rows = []
    for name, rids in groups.items():
        rows.append({"group": name, "subjects": len(rids),
                     "any T1 scan": len(rids & any_scan), "baseline T1 scan": len(rids & bl_scan)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n== Subjects with a baseline T1 scan, by phase ==")
    rows = {name: phase[list(rids & bl_scan)].value_counts() for name, rids in groups.items()}
    print(pd.DataFrame(rows).fillna(0).astype(int).to_string())


def _near_baseline(df: pd.DataFrame, first: pd.DataFrame, max_days: int) -> pd.DataFrame:
    """Keep each subject's row closest to their baseline diagnosis date, within max_days."""
    df = df.copy()
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df["DAYS"] = (df["EXAMDATE"] - df["RID"].map(first["EXAMDATE"])).dt.days.abs()
    df = df[df["DAYS"] <= max_days].sort_values("DAYS")
    return df.drop_duplicates("RID").set_index("RID")


def field_strength_and_freesurfer(folder: Path, first: pd.DataFrame, groups: dict,
                                  max_days: int = 90) -> None:
    """Count baseline field strength (scan information tables) and FreeSurfer coverage."""
    parts = []
    for name in ("MRIMETA", "MRI3META"):
        part = pd.read_csv(find_table(folder, name), low_memory=False)
        # Parse dates per table: the two tables do not use the same date format
        part["EXAMDATE"] = pd.to_datetime(part["EXAMDATE"], errors="coerce")
        parts.append(part)
    meta = pd.concat(parts, ignore_index=True)
    print("\n== MRI scan information tables ==")
    print("FIELD_STRENGTH values (all visits):",
          meta["FIELD_STRENGTH"].value_counts(dropna=False).to_dict())
    # Drop only scans marked "not conducted" (0). ADNI1 screening scans carry -4
    # (not recorded) although the scan exists, so they must be kept.
    meta = meta[meta["MMCONDCT"] != 0]
    base_meta = _near_baseline(meta.dropna(subset=["FIELD_STRENGTH"]), first, max_days)

    print(f"\n== Field strength of the scan nearest baseline (within {max_days} days) ==")
    rows = {name: base_meta.loc[base_meta.index.intersection(list(rids)), "FIELD_STRENGTH"]
            .value_counts() for name, rids in groups.items()}
    print(pd.DataFrame(rows).fillna(0).astype(int).to_string())

    fs = pd.read_csv(find_table(folder, "UCSFFSX7"), low_memory=False)
    print("\n== FreeSurfer 7.x table ==")
    print(f"{len(fs)} rows, {fs['RID'].nunique()} subjects, {fs.shape[1]} columns")
    print("OVERALLQC:", fs["OVERALLQC"].value_counts(dropna=False).to_dict())
    base_fs = _near_baseline(fs, first, max_days)
    passed = base_fs[base_fs["OVERALLQC"].astype(str).str.lower() == "pass"]
    rows = [{"group": name, "subjects": len(rids),
             "FreeSurfer at baseline": len(base_fs.index.intersection(list(rids))),
             "of which QC pass": len(passed.index.intersection(list(rids)))}
            for name, rids in groups.items()]
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    folder = Path(cfg["paths"]["raw_tables"])
    overview(folder)
    diagnosis_codes(folder)
    first, groups = baseline_counts(folder, cfg["progression"]["window_months"])
    scan_availability(folder, first, groups)
    field_strength_and_freesurfer(folder, first, groups)


if __name__ == "__main__":
    main()
