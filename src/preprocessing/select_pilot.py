"""Pick 10 scans for the preprocessing pilot (5 CN + 5 MCI, training subjects only).

Run from the repo root:  python -m src.preprocessing.select_pilot

Spreads the choice across ADNI phases and scan types. Writes data/tables/pilot_scans.csv and
data/tables/pilot_image_ids.txt (comma-separated, for the ADNI image search). Both files are
local only. Prints counts only.
"""
from pathlib import Path

import pandas as pd

TABLES = Path("data/tables")
SEED = 42
PER_COHORT = 5


def pick(cohort: pd.DataFrame, name: str) -> pd.DataFrame:
    """One scan per (phase, scan type) group first, then fill up at random."""
    cohort = cohort.assign(
        cohort=name,
        scan_type=cohort["image_description"].str.lower().map(
            lambda d: ("GE " if "spgr" in d else "") + ("accelerated" if "accel" in d else "standard")),
    )
    shuffled = cohort.sample(frac=1, random_state=SEED)
    one_per_group = shuffled.drop_duplicates(["BL_PHASE", "scan_type"])
    rest = shuffled.drop(one_per_group.index)
    return pd.concat([one_per_group, rest]).head(PER_COHORT)


def main() -> None:
    cn = pd.read_csv(TABLES / "cn_cohort.csv").merge(pd.read_csv(TABLES / "cn_splits.csv"), on="RID")
    mci = pd.read_csv(TABLES / "mci_cohort.csv").merge(pd.read_csv(TABLES / "mci_splits.csv"), on="RID")
    cn = cn[cn["is_baseline"] & (cn["split"] == "train")]
    mci = mci[mci["split"] == "train"]

    pilot = pd.concat([pick(cn, "CN"), pick(mci, "MCI")])
    cols = ["cohort", "RID", "PTID", "BL_PHASE", "image_id", "scan_date", "image_description",
            "scan_type"]
    pilot[cols].to_csv(TABLES / "pilot_scans.csv", index=False)
    (TABLES / "pilot_image_ids.txt").write_text(",".join(pilot["image_id"].astype(str)))

    print("pilot scans:", len(pilot), "| subjects:", pilot["RID"].nunique())
    print("by cohort and phase:", pilot.groupby(["cohort", "BL_PHASE"]).size().to_dict())
    print("by scan type:", pilot["scan_type"].value_counts().to_dict())
    print("written:", TABLES / "pilot_scans.csv", "and", TABLES / "pilot_image_ids.txt")


if __name__ == "__main__":
    main()
