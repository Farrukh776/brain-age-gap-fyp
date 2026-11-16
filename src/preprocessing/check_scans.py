"""Check that every essential scan is on disk and complete. Prints counts only.

Run from the repo root:  python -m src.preprocessing.check_scans

Essential scans = every MCI scan + every CN baseline scan in data/tables (same list as
make_download_lists.py, pilot scans included). For each one it checks that:
  - a folder I<image_id> exists under data/raw/scans
  - it holds DICOM files, none of them empty or cut off (smaller than MIN_BYTES)
  - the number of slices is plausible for a 3D T1 scan (at least MIN_SLICES)
Whether the slices form one readable series is tested by preprocess.py, which reads every
series in full and records any scan it cannot read as failed.
Scans found in more than one folder (repeated downloads) are compared and reported.
Writes the per-scan result to outputs/preprocessing/scan_check.csv (local only).
"""
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

TABLES = Path("data/tables")
RAW_SCANS = Path("data/raw/scans")
OUT = Path("outputs/preprocessing/scan_check.csv")
MIN_SLICES = 100
MIN_BYTES = 10_000  # one 256 x 256 slice is about 130 KB; anything tiny is cut off
# Newer scanners store the whole volume in one multi-frame file (about 22 MB for 170 slices)
MIN_SINGLE_FILE_BYTES = 10_000_000


def check_folder(folder: Path) -> tuple[str, int]:
    # Sizes come from the folder listing; opening every file is very slow on Windows.
    with os.scandir(folder) as entries:
        sizes = [e.stat().st_size for e in entries if e.name.lower().endswith(".dcm")]
    if not sizes:
        return "no DICOM files", 0
    if min(sizes) < MIN_BYTES:
        return "empty or cut-off file", len(sizes)
    if len(sizes) == 1 and sizes[0] >= MIN_SINGLE_FILE_BYTES:
        return "ok (one multi-frame file)", 1
    if len(sizes) < MIN_SLICES:
        return "too few slices", len(sizes)
    return "ok", len(sizes)


def main() -> None:
    mci = pd.read_csv(TABLES / "mci_cohort.csv")[["image_id"]].assign(cohort="MCI")
    cn = pd.read_csv(TABLES / "cn_cohort.csv")
    cn = cn[cn["is_baseline"]][["image_id"]].assign(cohort="CN")
    wanted = pd.concat([mci, cn]).drop_duplicates("image_id")

    folders = defaultdict(list)
    for p in RAW_SCANS.rglob("I*"):
        if p.is_dir() and p.name[1:].isdigit():
            folders[int(p.name[1:])].append(p)

    rows = []
    for number, (image_id, cohort) in enumerate(wanted.itertuples(index=False), 1):
        if number % 200 == 0:
            print(f"checked {number}/{len(wanted)}", flush=True)
        found = folders.get(image_id, [])
        if not found:
            rows.append({"image_id": image_id, "cohort": cohort, "status": "missing",
                         "n_files": 0, "copies": 0})
            continue
        results = [check_folder(f) for f in found]
        good = [r for r in results if r[0] == "ok"]
        status, n_files = good[0] if good else results[0]
        if len(good) > 1 and len({r[1] for r in good}) > 1:
            status = "copies differ"
        rows.append({"image_id": image_id, "cohort": cohort, "status": status,
                     "n_files": n_files, "copies": len(found)})

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)

    print("essential scans:", len(table))
    print(table.groupby(["cohort", "status"]).size().to_string())
    ok = table[table["status"] == "ok"]  # slice counts only for scans stored as one file per slice
    print("slices per scan (ok scans): min", ok["n_files"].min(), "| median",
          int(ok["n_files"].median()), "| max", ok["n_files"].max())
    print("scans present in more than one folder:", int((table["copies"] > 1).sum()))
    extra = set(folders) - set(table["image_id"])
    print("scan folders on disk not in the essential list:", len(extra))
    print("details:", OUT)


if __name__ == "__main__":
    main()
