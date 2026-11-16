"""Write lists of image IDs to paste into the ADNI Advanced Image Search.

Run from the repo root:
    python -m src.preprocessing.make_download_lists                  essential scans
    python -m src.preprocessing.make_download_lists --later-visits   later CN training visits

Essential scans: every MCI baseline scan and every CN baseline scan, written to
data/tables/essential_scans.csv. Later visits: the non-baseline scans of CN training subjects
who still have a usable baseline scan in data/tables/scan_qc_final.csv (the brain-age model
trains on all visits, see cohort_definition.md), written to data/tables/later_visit_scans.csv.
Either CSV is the input for preprocess.py. Scans whose folder already exists under
data/raw/scans are left out of the lists. Writes one comma-separated line per file to
data/tables/download_lists/ (local only). Prints counts only.
"""
import argparse
from pathlib import Path

import pandas as pd

TABLES = Path("data/tables")
RAW_SCANS = Path("data/raw/scans")
OUT = TABLES / "download_lists"
BATCH_SIZE = 200


def essential() -> list:
    mci = pd.read_csv(TABLES / "mci_cohort.csv")["image_id"]
    cn = pd.read_csv(TABLES / "cn_cohort.csv")
    cn = cn[cn["is_baseline"]]["image_id"]
    return pd.concat([mci, cn]).drop_duplicates().tolist()


def later_visits() -> list:
    cn = pd.read_csv(TABLES / "cn_cohort.csv")
    splits = pd.read_csv(TABLES / "cn_splits.csv").set_index("RID")["split"]
    final = pd.read_csv(TABLES / "scan_qc_final.csv")
    usable = set(final[(final["cohort"] == "CN") & (final["status"] != "excluded")]["RID"])
    later = cn[~cn["is_baseline"] & (cn["RID"].map(splits) == "train") & cn["RID"].isin(usable)]
    return later["image_id"].drop_duplicates().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--later-visits", action="store_true",
                        help="later CN training visits instead of the essential scans")
    args = parser.parse_args()
    name = "later_visit" if args.later_visits else "essential"
    wanted = later_visits() if args.later_visits else essential()
    pd.DataFrame({"image_id": wanted}).to_csv(TABLES / f"{name}_scans.csv", index=False)

    have = {int(p.name[1:]) for p in RAW_SCANS.rglob("I*") if p.is_dir() and p.name[1:].isdigit()}
    todo = [i for i in wanted if i not in have]

    prefix = "later" if args.later_visits else "batch"
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob(f"{prefix}_*.txt"):
        old.unlink()
    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    for number, batch in enumerate(batches, 1):
        (OUT / f"{prefix}_{number:02d}.txt").write_text(",".join(map(str, batch)))

    print(f"{name} scans:", len(wanted), "| already downloaded:", len(wanted) - len(todo),
          "| still to download:", len(todo))
    print("batch files:", len(batches), "of up to", BATCH_SIZE, "IDs each, in", OUT)


if __name__ == "__main__":
    main()
