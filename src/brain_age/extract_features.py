"""Extract voxel features for the SVR / ridge brain-age baseline (decision 22).

Run from the repo root:  python -m src.brain_age.extract_features --workers 4
Quick test on a few scans:  python -m src.brain_age.extract_features --limit 5

Each preprocessed scan (160 x 192 x 160 at 1 mm, z-scored) is shrunk to 4 mm by averaging
4 x 4 x 4 blocks, and only the blocks inside the template brain mask are kept (a block counts
as brain when at least half of its 1 mm voxels are brain). Every scan gives the same feature
columns in the same order, so the result is one matrix: rows = scans, columns = voxels.

No statistics are learned here (no PCA, no scaling), so nothing from validation or test can leak
into training. PCA and scaling are fitted later, on training folds only, inside the model
pipeline.

Scans: every scan the models use (CN training visits, CN val/test baselines, MCI baselines),
read through verify_processed.load_scans, so the list matches the final scan check.

Writes (local only, never committed):
  <features_dir>/voxels_4mm.npy        float32 matrix, one row per scan
  <features_dir>/voxels_4mm_index.csv  row order: image_id, cohort, split, RID, is_baseline
  <features_dir>/voxels_4mm_mask.npy   the 4 mm brain mask (to map features back to an image)
Prints counts only, never subject-level values.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

from src.preprocessing.preprocess import load_template
from src.preprocessing.verify_processed import load_scans

_W: dict = {}


def block_mean(array: np.ndarray, f: int) -> np.ndarray:
    """Average non-overlapping f x f x f blocks (every axis must divide by f)."""
    z, y, x = (s // f for s in array.shape)
    return array.reshape(z, f, y, f, x, f).mean(axis=(1, 3, 5))


def brain_mask_small(prep_cfg: dict, factor: int, min_fraction: float) -> np.ndarray:
    """Template brain mask cropped to the scan box, then shrunk to the feature grid."""
    template, mask, start = load_template(Path(prep_cfg["paths"]["template_dir"]),
                                          prep_cfg["output"]["shape"])
    box = sitk.RegionOfInterest(mask, prep_cfg["output"]["shape"], start)
    full = sitk.GetArrayFromImage(box).astype(np.float32)
    return block_mean(full, factor) >= min_fraction


def init(processed: str, mask: np.ndarray, factor: int) -> None:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    _W.update(processed=Path(processed), mask=mask, factor=factor)


def features(image_id: int) -> np.ndarray:
    path = _W["processed"] / f"{image_id}.nii.gz"
    array = sitk.GetArrayFromImage(sitk.ReadImage(str(path), sitk.sitkFloat32))
    small = block_mean(array.astype(np.float64), _W["factor"])
    return small[_W["mask"]].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/brain_age.yaml")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="only the first N scans (for testing)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    fcfg = cfg["features"]
    prep_cfg = yaml.safe_load(open(cfg["paths"]["preprocessing_config"]))
    factor = fcfg["downsample_factor"]
    shape = prep_cfg["output"]["shape"]
    if any(s % factor for s in shape):
        raise SystemExit(f"scan shape {shape} does not divide by {factor}")

    scans = load_scans()
    if args.limit:
        scans = scans.head(args.limit)
    mask = brain_mask_small(prep_cfg, factor, fcfg["mask_min_fraction"])
    print(f"{len(scans)} scans, {int(mask.sum())} features per scan "
          f"({factor} mm grid {mask.shape})")

    ids = scans["image_id"].tolist()
    init_args = (prep_cfg["paths"]["out_dir"], mask, factor)
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, initializer=init, initargs=init_args) as pool:
            rows = list(pool.map(features, ids, chunksize=8))
    else:
        init(*init_args)
        rows = [features(i) for i in ids]
    matrix = np.stack(rows)

    if not np.isfinite(matrix).all():
        raise SystemExit("NaN or inf in the features; check the processed scans")

    out = Path(cfg["paths"]["features_dir"])
    out.mkdir(parents=True, exist_ok=True)
    stem = "voxels_4mm" if not args.limit else "voxels_4mm_test"
    np.save(out / f"{stem}.npy", matrix)
    np.save(out / f"{stem}_mask.npy", mask)
    scans[["image_id", "cohort", "split", "RID", "is_baseline"]].to_csv(
        out / f"{stem}_index.csv", index=False)

    print(f"Saved {matrix.shape[0]} x {matrix.shape[1]} matrix "
          f"({matrix.nbytes / 1e6:.0f} MB) to {out / (stem + '.npy')}")
    print("Scans per cohort and split:")
    print(scans.groupby(["cohort", "split"]).size().to_string())


if __name__ == "__main__":
    main()
