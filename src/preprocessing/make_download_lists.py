"""Write lists of image IDs to paste into the ADNI Advanced Image Search.

Run from the repo root:  python -m src.preprocessing.make_download_lists

Covers the essential scans: every MCI baseline scan and every CN baseline scan. Scans whose
folder already exists under data/raw/scans are left out. Writes one comma-separated line per
file to data/tables/download_lists/ (local only). Prints counts only.
"""
from pathlib import Path

import pandas as pd

TABLES = Path("data/tables")
RAW_SCANS = Path("data/raw/scans")
OUT = TABLES / "download_lists"
BATCH_SIZE = 200


def main() -> None:
    mci = pd.read_csv(TABLES / "mci_cohort.csv")["image_id"]
    cn = pd.read_csv(TABLES / "cn_cohort.csv")
    cn = cn[cn["is_baseline"]]["image_id"]
    wanted = pd.concat([mci, cn]).drop_duplicates().tolist()

    have = {int(p.name[1:]) for p in RAW_SCANS.rglob("I*") if p.is_dir() and p.name[1:].isdigit()}
    todo = [i for i in wanted if i not in have]

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("batch_*.txt"):
        old.unlink()
    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    for number, batch in enumerate(batches, 1):
        (OUT / f"batch_{number:02d}.txt").write_text(",".join(map(str, batch)))

    print("essential scans:", len(wanted), "| already downloaded:", len(wanted) - len(todo),
          "| still to download:", len(todo))
    print("batch files:", len(batches), "of up to", BATCH_SIZE, "IDs each, in", OUT)


if __name__ == "__main__":
    main()
