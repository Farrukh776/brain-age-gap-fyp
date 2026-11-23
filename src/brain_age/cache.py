"""Build, check and move the 2 mm cache that the 3D CNN reads (~6 GB instead of 30 GB of scans).

On the laptop (reads data/processed):
    python -m src.brain_age.cache build --cache-dir D:/fyp-data/cache --workers 4
    python -m src.brain_age.cache pack  --cache-dir D:/fyp-data/cache --zip D:/fyp-data/fyp_cache_2mm.zip
Upload the zip to the university Google Drive (decision 27). On the other machine, in the repo:
    python -m src.brain_age.cache unpack --zip <downloaded zip> --cache-dir <folder> --workers 4
Then train with:  python -m src.brain_age.train_cnn --run cnn-1 --cache-dir <folder>

build:  shrinks every scan the models use (all 2,489: CN train / val / test and MCI) to 2 mm with
        the same code as training (data.BrainAgeDataset.load), saves <cache-dir>/2mm/<image_id>.npy
        and writes <cache-dir>/manifest.csv: size and SHA-256 checksum of every cached array and
        of the tables the CNN reads. Arrays already in the cache are reused, not rebuilt.
check:  recomputes every checksum and compares with the manifest (catches missing, cut-short or
        changed files), confirms every scan in the tables has a cached array, and opens each array
        (shape, float32, no NaN / inf). Read-only. Prints counts and file names only.
pack:   one zip holding cache/2mm/*.npy, cache/manifest.csv and tables/*.csv (one file uploads
        far more reliably than 2,495).
unpack: extracts the cache to --cache-dir and the tables to data/tables (refuses to overwrite a
        different table), then runs check.
The cache holds ADNI-derived data: never committed; stored only locally or on the university
Google Drive, shared only with team members who have ADNI access (decisions 2, 14, 27).
"""
import argparse
import hashlib
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

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


def build(cfg: dict, root: Path, factor: int, workers: int, limit: int | None) -> None:
    prep = yaml.safe_load(open(cfg["paths"]["preprocessing_config"]))
    table = scan_table(cfg)
    if limit:
        table = table.head(limit)
    table = table.reset_index(drop=True)
    print(f"Building {factor} mm cache for {len(table)} scans in {root / f'{factor}mm'}")
    init_args = (table, prep["paths"]["out_dir"], factor, str(root))
    with ProcessPoolExecutor(workers, initializer=init_build, initargs=init_args) as pool:
        rows = list(pool.map(build_one, range(len(table)), chunksize=8))
    rows += [(f"tables/{t}", (TABLES / t).stat().st_size, sha256(TABLES / t))
             for t in TABLE_FILES]
    manifest = pd.DataFrame(rows, columns=["file", "bytes", "sha256"])
    manifest.to_csv(root / "manifest.csv", index=False)
    print(f"Wrote {root / 'manifest.csv'}: {len(manifest) - len(TABLE_FILES)} arrays + "
          f"{len(TABLE_FILES)} tables, {manifest['bytes'].sum() / 1e9:.1f} GB")


def check(cfg: dict, root: Path, factor: int, workers: int) -> None:
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    jobs = []
    for f, b, h in manifest.itertuples(index=False):
        base = TABLES.parent if f.startswith("tables/") else root
        jobs.append((str(base), f, int(b), h))
    with ProcessPoolExecutor(workers) as pool:
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


def pack(root: Path, zip_path: Path) -> None:
    manifest = pd.read_csv(root / "manifest.csv")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(root / "manifest.csv", "cache/manifest.csv")
        for f in manifest["file"]:
            if f.startswith("tables/"):
                z.write(TABLES / f.split("/", 1)[1], f)
            else:
                z.write(root / f, f"cache/{f}")
    print(f"Wrote {zip_path}: {len(manifest) + 1} files, "
          f"{zip_path.stat().st_size / 1e9:.2f} GB (from {manifest['bytes'].sum() / 1e9:.2f} GB)")


def unpack(zip_path: Path, root: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for n in names:  # only the layout pack writes; nothing outside the targets
            p = PurePosixPath(n)
            if p.is_absolute() or ".." in p.parts or p.parts[0] not in ("cache", "tables"):
                raise SystemExit(f"unexpected entry in zip: {n}")
        for n in names:
            if n.startswith("tables/"):
                target = TABLES / n.split("/", 1)[1]
                if target.exists():
                    if target.read_bytes() != z.read(n):
                        raise SystemExit(f"{target} exists and differs; move it away first")
                    continue
            else:
                target = root / n.split("/", 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(target, "wb") as dst:
                while block := src.read(1 << 20):
                    dst.write(block)
    print(f"Extracted {len(names)} files: cache to {root}, tables to {TABLES}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["build", "check", "pack", "unpack"])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--zip", help="pack / unpack: path of the zip file")
    parser.add_argument("--config", default="configs/brain_age.yaml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="build only: first N scans (testing)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    factor = int(cfg["cnn"]["input_downsample_mm"])
    root = Path(args.cache_dir)
    if args.action in ("pack", "unpack") and not args.zip:
        raise SystemExit("--zip is required for pack / unpack")

    if args.action == "build":
        build(cfg, root, factor, args.workers, args.limit)
        print("Next: 'check', then 'pack' to make one zip for the university Google Drive.")
    elif args.action == "check":
        check(cfg, root, factor, args.workers)
    elif args.action == "pack":
        pack(root, Path(args.zip))
    else:
        unpack(Path(args.zip), root)
        check(cfg, root, factor, args.workers)


if __name__ == "__main__":
    main()
