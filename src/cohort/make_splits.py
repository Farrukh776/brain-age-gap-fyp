"""Create the subject-level train / validation / test splits for both cohorts.

Run from the repo root:  python -m src.cohort.make_splits

Reads data/tables/cn_cohort.csv and mci_cohort.csv, writes cn_splits.csv and mci_splits.csv
(one row per subject: RID, split). Split files are created once: the script refuses to
overwrite them. Prints counts only.

- MCI subjects are stratified by label (stable / progressor).
- CN subjects are stratified by age band, so each split covers the full age range.
"""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

CONFIG = Path("configs/cohort.yaml")


def split_subjects(subjects: pd.DataFrame, strata: pd.Series, cfg: dict, seed: int) -> pd.Series:
    """Return a Series (index = RID) with 'train', 'val' or 'test' for every subject."""
    rest_share = cfg["val"] + cfg["test"]
    train, rest = train_test_split(subjects["RID"], test_size=rest_share,
                                   stratify=strata, random_state=seed)
    val, test = train_test_split(rest, test_size=cfg["test"] / rest_share,
                                 stratify=strata.loc[rest.index], random_state=seed)
    split = pd.Series("train", index=subjects["RID"], name="split")
    split[val.values] = "val"
    split[test.values] = "test"
    return split


def check_no_overlap(split: pd.Series) -> None:
    assert split.index.is_unique, "A subject appears twice"
    groups = [set(split[split == name].index) for name in ("train", "val", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sum(len(g) for g in groups) == len(split)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    out_dir, seed = Path(cfg["paths"]["out_dir"]), cfg["seed"]
    cn_file, mci_file = out_dir / "cn_splits.csv", out_dir / "mci_splits.csv"
    if cn_file.exists() or mci_file.exists():
        raise SystemExit("Split files already exist. They are created once and never "
                         "regenerated (see docs/cohort_definition.md).")

    # ---- progression cohort: one row per subject, stratified by label
    mci = pd.read_csv(out_dir / "mci_cohort.csv")
    mci_split = split_subjects(mci, mci["label"], cfg["split"], seed)
    check_no_overlap(mci_split)
    mci = mci.join(mci_split, on="RID")
    print("== Progression cohort ==")
    table = pd.crosstab(mci["split"], mci["label"], margins=True)
    table.columns = ["stable", "progressor", "all"]
    print(table.loc[["train", "val", "test", "All"]].to_string())
    print("progressor share:", mci.groupby("split")["label"].mean().round(3).to_dict())
    print("mean age:", mci.groupby("split")["age"].mean().round(1).to_dict())

    # ---- brain-age cohort: split subjects (not scans), stratified by baseline age band
    cn = pd.read_csv(out_dir / "cn_cohort.csv")
    cn_subjects = cn[cn["is_baseline"]].reset_index(drop=True)
    age_band = pd.qcut(cn_subjects["age"], q=5, labels=False)
    cn_split = split_subjects(cn_subjects, age_band, cfg["split"], seed)
    check_no_overlap(cn_split)
    cn = cn.join(cn_split, on="RID")
    base = cn[cn["is_baseline"]]
    print("\n== Brain-age cohort ==")
    print("subjects:", base["split"].value_counts().to_dict())
    print("scans (all visits):", cn["split"].value_counts().to_dict())
    print("mean baseline age:", base.groupby("split")["age"].mean().round(1).to_dict())
    print("age range:", base.groupby("split")["age"].agg(["min", "max"]).round(0).to_dict())
    print("female share:", (base["sex"] == "F").groupby(base["split"]).mean().round(2).to_dict())

    both = set(cn_split.index) & set(mci_split.index)
    assert not both, "A subject is in both cohorts"
    print("\nsubjects in more than one split or cohort: 0")

    cn_split.reset_index().to_csv(cn_file, index=False)
    mci_split.reset_index().to_csv(mci_file, index=False)
    print("written:", cn_file, "and", mci_file)


if __name__ == "__main__":
    main()
