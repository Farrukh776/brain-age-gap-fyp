"""Find processed scans that are tilted relative to the rest (decision 21).

Run from the repo root:  python -m src.preprocessing.check_tilt --workers 6   (about 70 min)

Builds a study mean image from every scan the models use (all are already in MNI space), then
registers each scan to that mean (affine, correlation) and measures the rotation needed.
Aligned scans need at most about 2.5 degrees; confirmed tilts needed 3.7 to 14 degrees.
Millimetres of movement are not used: the affine also removes normal head-size differences.

Writes outputs/preprocessing/tilt_check.csv (local only: image_id, rotation, scales) and
prints the image IDs above --max-rotation. Redo those with
`python -m src.preprocessing.preprocess --scans <list> --retry-wide` after backing up their
files, keep a new version only if it is less rotated and passes an eye check.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.preprocessing.verify_processed import load_scans

PROCESSED = Path("data/processed")
OUT = Path("outputs/preprocessing/tilt_check.csv")
_W: dict = {}


def load(image_id: int) -> sitk.Image:
    return sitk.ReadImage(str(PROCESSED / f"{image_id}.nii.gz"), sitk.sitkFloat32)


def partial_sum(ids: list) -> np.ndarray:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    total = None
    for i in ids:
        a = sitk.GetArrayFromImage(load(i)).astype(np.float64)
        total = a if total is None else total + a
    return total


def init(mean_array: np.ndarray, reference_id: int) -> None:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    mean = sitk.GetImageFromArray(mean_array.astype(np.float32))
    mean.CopyInformation(load(reference_id))
    _W["mean"] = mean


def rotation(image_id: int) -> tuple:
    fixed, moving = _W["mean"], load(image_id)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, 42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(1.0, 0.001, 200, 0.6)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2])
    reg.SetSmoothingSigmasPerLevel([2, 1])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    tx = sitk.AffineTransform(3)
    tx.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(
        [s / 2 for s in fixed.GetSize()]))
    reg.SetInitialTransform(tx, inPlace=True)
    reg.Execute(fixed, moving)
    u, scales, vt = np.linalg.svd(np.reshape(tx.GetMatrix(), (3, 3)))
    rot = u @ vt  # closest pure rotation
    angle = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    return image_id, round(float(angle), 2), round(float(scales.min()), 3), \
        round(float(scales.max()), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-rotation", type=float, default=4.0)
    args = ap.parse_args()

    ids = load_scans()["image_id"].tolist()
    with ProcessPoolExecutor(args.workers) as ex:
        parts = list(ex.map(partial_sum, [ids[k::args.workers] for k in range(args.workers)]))
    mean = sum(parts) / len(ids)
    print(f"study mean built from {len(ids)} scans", flush=True)

    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(mean, ids[0])) as ex:
        rows = list(ex.map(rotation, ids, chunksize=8))
    res = pd.DataFrame(rows, columns=["image_id", "rotation_deg", "scale_min", "scale_max"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    q = res["rotation_deg"].quantile([0.5, 0.95, 0.99]).round(2).tolist()
    tilted = res[res["rotation_deg"] > args.max_rotation].sort_values("rotation_deg")
    print(f"rotation median / 95% / 99%: {q} degrees")
    print(f"above {args.max_rotation} degrees: {len(tilted)}  -> {tilted['image_id'].tolist()}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
