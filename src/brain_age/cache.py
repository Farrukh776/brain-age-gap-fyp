"""Build and check the 2 mm cache that the 3D CNN reads (so the lab GPU needs ~6 GB, not 30 GB).

On the laptop (reads data/processed, about 20 min):
    python -m src.brain_age.cache build --cache-dir D:/fyp-data/cache --workers 4
After copying to another machine (cache folder + data/tables):
    python -m src.brain_age.cache check --cache-dir D:/fyp-cache --workers 4
Then train with:  python -m src.brain_age.train_cnn --run cnn-1 --cache-dir D:/fyp-cache

build: shrinks every scan the models use (all 2,489: CN train / val / test and MCI) to 2 mm with
       the same code as training (data.BrainAgeDataset.load), saves <cache-dir>/2mm/<image_id>.npy
       and writes <cache-dir>/manifest.csv: size and SHA-256 checksum of every cached array and of
       the tables the CNN reads. Arrays already in the cache are reused, not rebuilt.
check: recomputes every checksum and compares with the manifest (catches missing, cut-short or
       changed files), confirms every scan in the tables has a cached array, and opens each array
       (shape, float32, no NaN / inf). Read-only. Prints counts and file names only.
The cache holds ADNI-derived data: local only, never committed or uploaded (decisions 2, 14).
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.brain_age.data import BrainAgeDataset, scan_table

TABLES = Path("data/tables")
TABLE_FILES = ["cn_train_scans.csv", "scan_qc_final.csv", "cn_cohort.csv", "mci_cohort.csv",
               "cn_splits.csv", "mci_splits.csv"]
SHAPE = (80, 96, 80)
_W: dict = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def init_build(table, processed, factor, cache_dir):
    _W["ds"] = BrainAgeDataset(table, processed, factor, cache_dir=cache_dir)


def build_one(i: int) -> tuple:
    ds = _W["ds"]
    ds.load(i)  # writes the cached array if it is not there yet
    path = ds.cache / f"{ds.table.at[i, 'image_id']}.npy"
    return f"{ds.cache.name}/{path.name}", path.stat().st_size, sha256(path)


def check_one(args: tuple) -> list:
    root, name, size, digest = args
    path = Path(root) / name
    if not path.exists():
        return [f"missing: {name}"]
    problems = []
    if path.stat().st_size != size:
        problems.append(f"wrong size: {name}")
    elif sha256(path) != digest:
        problems.append(f"checksum differs: {name}")
    if name.endswith(".npy") and not problems:
        a = np.load(path)
        if a.shape != SHAPE or a.dtype != np.float32 or not np.isfinite(a).all():
            problems.append(f"bad array: {name}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["build", "check"])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--config", default="configs/brain_age.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="build only: first N scans (testing)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    factor = int(cfg["cnn"]["input_downsample_mm"])
    root = Path(args.cache_dir)
    manifest_path = root / "manifest.csv"

    if args.action == "build":
        prep = yaml.safe_load(open(cfg["paths"]["preprocessing_config"]))
        table = scan_table(cfg)
        if args.limit:
            table = table.head(args.limit)
        table = table.reset_index(drop=True)
        print(f"Building {factor} mm cache for {len(table)} scans in {root / f'{factor}mm'}")
        init_args = (table, prep["paths"]["out_dir"], factor, str(root))
        with ProcessPoolExecutor(args.workers, initializer=init_build,
                                 initargs=init_args) as pool:
            rows = list(pool.map(build_one, range(len(table)), chunksize=8))
        rows += [(f"tables/{t}", (TABLES / t).stat().st_size, sha256(TABLES / t))
                 for t in TABLE_FILES]
        manifest = pd.DataFrame(rows, columns=["file", "bytes", "sha256"])
        manifest.to_csv(manifest_path, index=False)
        gb = manifest["bytes"].sum() / 1e9
        print(f"Wrote {manifest_path}: {len(manifest) - len(TABLE_FILES)} arrays + "
              f"{len(TABLE_FILES)} tables, {gb:.1f} GB")
        print(f"Copy the folder {root} and data/tables/ to the other machine, then run 'check'.")
        return

    # check
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    jobs = []
    for f, b, h in manifest.itertuples(index=False):
        base = TABLES.parent if f.startswith("tables/") else root
        jobs.append((str(base), f, int(b), h))
    with ProcessPoolExecutor(args.workers) as pool:
        problems = [p for ps in pool.map(check_one, jobs, chunksize=16) for p in ps]

    # Every scan the models use must have a cached array (tables checked above first)
    if not any("tables/" in p for p in problems):
        table = scan_table(cfg)
        listed = set(manifest["file"])
        missing = [i for i in table["image_id"] if f"{factor}mm/{i}.npy" not in listed]
        if missing:
            problems.append(f"{len(missing)} scans in the tables have no cached array")
        print("Scans per cohort and split:\n"
              + table.groupby(["cohort", "split"]).size().to_string())

    n_arrays = int(manifest["file"].str.endswith(".npy").sum())
    print(f"Checked {n_arrays} arrays and {len(manifest) - n_arrays} tables against the manifest")
    if problems:
        print(f"PROBLEMS ({len(problems)}):\n  " + "\n  ".join(problems[:30]))
        if len(problems) > 30:
            print(f"  ... and {len(problems) - 30} more")
        raise SystemExit(1)
    print("All files match the manifest. Ready to train.")


if __name__ == "__main__":
    main()
