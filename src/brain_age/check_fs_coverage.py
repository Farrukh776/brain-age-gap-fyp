"""Check how many of our scans have a row in ADNI's FreeSurfer 7.x table (UCSFFSX7).

Run from the repo root:  python -m src.brain_age.check_fs_coverage

Read-only. Prints counts per group only (no subject rows). A scan is matched
  - "exact" if its image_id equals the table's IMAGEUID, or
  - "same session" if not exact but the table has a row for the same RID with EXAMDATE
    within `--days` days of our scan date (FreeSurfer was run on another T1 of that visit).
For matched scans it also counts the FreeSurfer QC result (OVERALLQC; blank = not rated).
"""
import argparse

import pandas as pd
import yaml


def load_scans(paths: dict) -> pd.DataFrame:
    """One row per scan we use: group, RID, image_id, scan_date."""
    train = pd.read_csv(paths["train_scans"])
    train["group"] = train["is_baseline"].map({True: "CN train baseline",
                                               False: "CN train later"})

    qc = pd.read_csv(paths["scan_qc"])
    qc = qc[qc["status"] != "excluded"].copy()
    qc = qc[~((qc["cohort"] == "CN") & (qc["split"] == "train"))]  # already in train
    qc["image_id"] = qc["image_id"].astype("int64")
    qc["original_image_id"] = qc["original_image_id"].astype("int64")
    qc["group"] = qc["cohort"] + " " + qc["split"]

    cohort = pd.concat([pd.read_csv(paths["cohort_table"]),
                        pd.read_csv(paths["mci_cohort_table"])])
    dates = cohort.drop_duplicates("image_id").set_index("image_id")["scan_date"]
    # A replacement scan comes from the same session as the original, so the date is the same.
    qc["scan_date"] = qc["image_id"].map(dates).fillna(qc["original_image_id"].map(dates))

    cols = ["group", "RID", "image_id", "scan_date"]
    scans = pd.concat([train[cols], qc[cols]], ignore_index=True)
    scans["scan_date"] = pd.to_datetime(scans["scan_date"])
    assert scans["scan_date"].notna().all(), "scan date missing"
    assert scans["image_id"].is_unique
    return scans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/brain_age.yaml")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    paths = yaml.safe_load(open(args.config))["paths"]

    scans = load_scans(paths)
    fs = pd.read_csv(paths["freesurfer_table"], low_memory=False,
                     usecols=["RID", "IMAGEUID", "EXAMDATE", "OVERALLQC", "FIELD_STRENGTH"])
    fs["EXAMDATE"] = pd.to_datetime(fs["EXAMDATE"])
    fs["OVERALLQC"] = fs["OVERALLQC"].fillna("not rated")

    by_uid = fs.drop_duplicates("IMAGEUID").set_index("IMAGEUID")
    scans["match"] = "none"
    scans["fs_qc"] = None
    exact = scans["image_id"].isin(by_uid.index)
    scans.loc[exact, "match"] = "exact"
    scans.loc[exact, "fs_qc"] = scans.loc[exact, "image_id"].map(by_uid["OVERALLQC"])

    # Same-session fallback: nearest FreeSurfer row of the same subject within --days.
    rest = scans[~exact].reset_index()
    near = pd.merge_asof(rest.sort_values("scan_date"),
                         fs.sort_values("EXAMDATE")[["RID", "EXAMDATE", "OVERALLQC"]],
                         left_on="scan_date", right_on="EXAMDATE", by="RID",
                         direction="nearest", tolerance=pd.Timedelta(days=args.days))
    hit = near[near["EXAMDATE"].notna()]
    scans.loc[hit["index"], "match"] = "same session"
    scans.loc[hit["index"], "fs_qc"] = hit["OVERALLQC"].values

    order = ["CN train baseline", "CN train later", "CN val", "CN test",
             "MCI train", "MCI val", "MCI test"]
    table = pd.crosstab(scans["group"], scans["match"]).reindex(order).fillna(0).astype(int)
    for c in ["exact", "same session", "none"]:
        if c not in table:
            table[c] = 0
    table = table[["exact", "same session", "none"]]
    table["total"] = table.sum(axis=1)
    table["covered %"] = (100 * (table["exact"] + table["same session"])
                          / table["total"]).round(1)
    pd.set_option("display.width", 120)
    print(f"FreeSurfer rows: {len(fs)} | same-session window: {args.days} days\n")
    print(table.to_string(), "\n")

    matched = scans[scans["match"] != "none"]
    qc = pd.crosstab(matched["group"], matched["fs_qc"]).reindex(order).fillna(0).astype(int)
    print("FreeSurfer OVERALLQC among matched scans:")
    print(qc.to_string())


if __name__ == "__main__":
    main()
