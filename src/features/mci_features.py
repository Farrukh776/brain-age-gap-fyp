"""Build the MRI feature table for the MCI progression cohort (step 8, decision 33).

Run from the repo root:  python -m src.features.mci_features

E1-E5 all use the same MRI features: the 4 mm voxel features of decision 22 (29,520 brain
blocks per scan, `src/brain_age/extract_features.py`), one baseline scan per MCI subject. This
script only copies those rows out of the full voxel matrix; it learns no statistics (no scaling,
no PCA), so nothing from validation or test can leak. Scaling and PCA are fitted later, on
training folds only, inside the model pipeline.

Checks: one row per subject; every subject is in mci_cohort.csv with a label; splits match
mci_splits.csv; the scan is the one used for the BAG table (same subjects, same image IDs);
the features are finite.

Writes (local only, never committed; ADNI-derived):
  <features_dir>/mci_voxels_4mm.npy        float32, one row per MCI subject
  <features_dir>/mci_voxels_4mm_index.csv  row order: RID, image_id, split
Labels are not stored here; experiment code joins them from mci_cohort.csv.
Prints counts only, never subject-level values.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/brain_age.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    feat_dir = Path(cfg["paths"]["features_dir"])
    cohort_path = Path(cfg["paths"]["mci_cohort_table"])

    index = pd.read_csv(feat_dir / "voxels_4mm_index.csv")
    X = np.load(feat_dir / "voxels_4mm.npy", mmap_mode="r")
    if len(index) != X.shape[0]:
        raise SystemExit("voxel matrix and its index have different row counts")

    rows = index[(index["cohort"] == "MCI") & index["is_baseline"]].copy()
    rows["row"] = rows.index
    if rows["RID"].duplicated().any():
        raise SystemExit("more than one MCI baseline scan for a subject")

    cohort = pd.read_csv(cohort_path)
    splits = pd.read_csv(cohort_path.with_name("mci_splits.csv"))
    t = rows.merge(cohort[["RID", "label"]], on="RID", how="left", validate="one_to_one")
    t = t.merge(splits.rename(columns={"split": "split_table"}), on="RID", how="left",
                validate="one_to_one")
    if t["label"].isna().any():
        raise SystemExit("some feature rows are not in the MCI cohort or have no label")
    if (t["split"] != t["split_table"]).any():
        raise SystemExit("split differs from mci_splits.csv")

    # Same scans as the BAG table, so E1-E5 and BAG describe the same image of each subject
    bag = pd.read_csv(Path(cfg["paths"]["out_dir"]) / "bag" / "bag_table.csv")
    same = t[["RID", "image_id", "split"]].merge(bag[["RID", "image_id", "split"]],
                                                  how="outer", indicator=True)
    if (same["_merge"] != "both").any():
        raise SystemExit("subjects, scans or splits differ from the BAG table")

    t = t.sort_values(["split", "RID"]).reset_index(drop=True)
    features = np.asarray(X[t["row"].to_numpy()], dtype=np.float32)
    if not np.isfinite(features).all():
        raise SystemExit("features contain NaN or infinite values")

    np.save(feat_dir / "mci_voxels_4mm.npy", features)
    t[["RID", "image_id", "split"]].to_csv(feat_dir / "mci_voxels_4mm_index.csv", index=False)

    counts = t.groupby(["split", "label"]).size().unstack().rename(
        columns={0: "stable", 1: "progressor"})
    print(f"MCI feature table: {features.shape[0]} subjects x {features.shape[1]} voxel features "
          f"(cohort {len(cohort)}; {len(cohort) - len(t)} removed at QC)")
    print(counts.to_string())
    print(f"Matches the BAG table (same subjects, scans and splits). Saved "
          f"{feat_dir / 'mci_voxels_4mm.npy'}")


if __name__ == "__main__":
    main()
