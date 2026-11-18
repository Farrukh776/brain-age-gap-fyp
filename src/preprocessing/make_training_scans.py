"""Build the list of CN scans the brain-age model trains on (all visits), after scan QC.

Run from the repo root:  python -m src.preprocessing.make_training_scans

Combines the baseline scan of every usable CN training subject (data/tables/scan_qc_final.csv)
with that subject's later visits that passed the visual check
(outputs/preprocessing/visual_review.csv, verdict "ok"). A later visit that failed QC is
dropped; the subject keeps its other scans. Writes data/tables/cn_train_scans.csv (local only)
with one row per scan: RID, image_id, scan_date, age, is_baseline. Prints counts only.
"""
from pathlib import Path

import pandas as pd

TABLES = Path("data/tables")
REVIEW = Path("outputs/preprocessing/visual_review.csv")


def main() -> None:
    cohort = pd.read_csv(TABLES / "cn_cohort.csv")
    final = pd.read_csv(TABLES / "scan_qc_final.csv")
    review = pd.read_csv(REVIEW).set_index("image_id")["verdict"]

    usable = final[(final["cohort"] == "CN") & (final["split"] == "train")
                   & (final["status"] != "excluded")]
    baseline = cohort[cohort["image_id"].isin(usable["image_id"])]
    later = cohort[~cohort["is_baseline"] & cohort["RID"].isin(usable["RID"])]
    passed = later[later["image_id"].map(review) == "ok"]

    scans = pd.concat([baseline.assign(is_baseline=True), passed])
    cols = ["RID", "image_id", "scan_date", "age", "is_baseline"]
    scans = scans[cols].sort_values(["RID", "scan_date"])
    assert scans["image_id"].is_unique
    scans.to_csv(TABLES / "cn_train_scans.csv", index=False)

    print("training subjects:", scans["RID"].nunique(),
          "| baseline scans:", int(scans["is_baseline"].sum()),
          "| later visits kept:", len(passed), "of", len(later),
          "| total scans:", len(scans))
    print("age: min %.1f, mean %.1f, max %.1f" % (
        scans["age"].min(), scans["age"].mean(), scans["age"].max()))


if __name__ == "__main__":
    main()
