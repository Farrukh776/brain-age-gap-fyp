"""Show one scan before and after preprocessing, side by side.

Run from the repo root:
    python -m src.preprocessing.view_scan                 # a random preprocessed scan
    python -m src.preprocessing.view_scan --image-id 123  # a chosen scan

Top row: the raw DICOM scan as downloaded (middle slice in each direction).
Bottom row: the preprocessed scan from data/processed, with the template brain outline in red.
The picture is saved to outputs/preprocessing/compare/<image_id>.png (local only) and shown
in a window. Close the window to finish.
"""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import yaml

from src.preprocessing.preprocess import load_template, read_dicom

CONFIG = Path("configs/preprocessing.yaml")


def middle_slices(array: np.ndarray) -> list:
    """Sagittal, coronal and axial middle slices of a (z, y, x) array."""
    z, y, x = (s // 2 for s in array.shape)
    return [array[:, :, x], array[:, y, :], array[z, :, :]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", type=int, help="leave out for a random processed scan")
    parser.add_argument("--no-window", action="store_true", help="only save the picture")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    raw_dir, out_dir = Path(cfg["paths"]["raw_scans"]), Path(cfg["paths"]["out_dir"])
    save_dir = Path(cfg["paths"]["qc_dir"]) / "compare"
    save_dir.mkdir(parents=True, exist_ok=True)

    image_id = args.image_id
    if image_id is None:
        done = sorted(out_dir.glob("*.nii.gz"))
        if not done:
            raise SystemExit("no preprocessed scans in " + str(out_dir))
        image_id = int(random.choice(done).name.split(".")[0])

    folders = list(raw_dir.rglob(f"I{image_id}"))
    processed_file = out_dir / f"{image_id}.nii.gz"
    if not folders:
        raise SystemExit(f"raw scan folder I{image_id} not found under {raw_dir}")
    if not processed_file.exists():
        raise SystemExit(f"{processed_file} does not exist yet; run preprocess.py first")

    raw = read_dicom(folders[0], cfg["steps"]["orient"])
    raw_array = sitk.GetArrayFromImage(raw)
    processed = sitk.GetArrayFromImage(sitk.ReadImage(str(processed_file)))
    _, mask, start = load_template(Path(cfg["paths"]["template_dir"]), cfg["output"]["shape"])
    brain = sitk.GetArrayFromImage(
        sitk.RegionOfInterest(mask, cfg["output"]["shape"], start)) > 0

    if not args.no_window:
        plt.switch_backend("TkAgg")  # preprocess.py sets a file-only backend on import

    lo, hi = np.percentile(raw_array, [0.5, 99.5])
    names = ["sagittal (side)", "coronal (front)", "axial (top)"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, img, name in zip(axes[0], middle_slices(raw_array), names):
        ax.imshow(img, cmap="gray", origin="lower", vmin=lo, vmax=hi, aspect="auto")
        ax.set_title("raw: " + name)
    for ax, img, outline, name in zip(axes[1], middle_slices(processed),
                                      middle_slices(brain.astype(float)), names):
        ax.imshow(img, cmap="gray", origin="lower", vmin=-2.5, vmax=2.5)
        ax.contour(outline, levels=[0.5], colors="red", linewidths=0.8)
        ax.set_title("preprocessed: " + name)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"image {image_id}: raw {raw.GetSize()} voxels, "
                 f"{[round(s, 2) for s in raw.GetSpacing()]} mm  ->  "
                 f"preprocessed {processed.shape[::-1]} voxels, 1 mm")
    fig.tight_layout()
    path = save_dir / f"{image_id}.png"
    fig.savefig(path, dpi=90)
    print("saved", path)
    if not args.no_window:
        plt.show()


if __name__ == "__main__":
    main()
