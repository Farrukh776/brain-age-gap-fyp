"""Build the BAG table for the MCI progression cohort (step 7).

    python -m src.brain_age.predict --set mci      # first: frozen model + correction (decision 31)
    python -m src.brain_age.bag_table

Joins the MCI predictions (outputs/brain_age/final/mci_predictions.csv) with the cohort labels
(mci_cohort.csv: 1 = dementia within 36 months, 0 = stable; cohort_definition.md) and splits.
One row per subject:
    RID, image_id, split, age, brain_age (predicted), bag_raw (= brain_age - age),
    bag (bias-corrected, the BAG used from now on), label
Checks: one baseline scan per subject, every subject has a label and the split of mci_splits.csv,
the scan is the cohort's baseline scan (or its QC replacement), and labels per split match the
cohort counts. Writes outputs/brain_age/bag/bag_table.csv (local only; ADNI-derived).
Prints counts only, no subject rows.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/brain_age.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["paths"]["out_dir"])

    pred = pd.read_csv(out / "final" / "mci_predictions.csv")
    cohort = pd.read_csv(cfg["paths"]["mci_cohort_table"])
    splits = pd.read_csv(Path(cfg["paths"]["mci_cohort_table"]).with_name("mci_splits.csv"))

    if pred["RID"].duplicated().any():
        raise SystemExit("more than one scan for a subject")
    t = pred.merge(cohort[["RID", "image_id", "label"]].rename(columns={"image_id": "cohort_image"}),
                   on="RID", how="left", validate="one_to_one")
    t = t.merge(splits.rename(columns={"split": "split_table"}), on="RID", how="left",
                validate="one_to_one")
    if t["label"].isna().any():
        raise SystemExit("some subjects have no label")
    if (t["split"] != t["split_table"]).any():
        raise SystemExit("split differs from mci_splits.csv")
    # Scan must be the cohort baseline scan, or the QC replacement of it (decision 17)
    qc = pd.read_csv(cfg["paths"]["scan_qc"])
    replaced = dict(zip(qc["image_id"], qc.get("original_image_id", qc["image_id"])))
    original = t["image_id"].map(lambda i: replaced.get(i, i))
    same = (t["image_id"] == t["cohort_image"]) | (original == t["cohort_image"])
    if not same.all():
        raise SystemExit(f"{(~same).sum()} subjects use a scan other than their baseline scan")

    table = t[["RID", "image_id", "split", "age", "brain_age", "bag_raw", "bag", "label"]]
    table = table.astype({"label": int}).sort_values(["split", "RID"])
    dest = out / "bag"
    dest.mkdir(parents=True, exist_ok=True)
    table.to_csv(dest / "bag_table.csv", index=False)

    n_replaced = int((t["image_id"] != t["cohort_image"]).sum())
    counts = table.groupby(["split", "label"]).size().unstack().rename(
        columns={0: "stable", 1: "progressor"})
    print(f"BAG table: {len(table)} subjects ({n_replaced} use a QC replacement scan of the same "
          f"session); not in table: {len(cohort) - len(table)} cohort subject(s) removed at QC")
    print(counts.to_string())
    print(f"Saved {dest / 'bag_table.csv'}")


if __name__ == "__main__":
    main()
