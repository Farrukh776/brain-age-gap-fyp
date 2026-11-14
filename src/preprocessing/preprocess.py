"""Preprocess T1 scans: DICOM -> oriented, bias-corrected, MNI-aligned, cropped, normalised.

Run from the repo root, for example:
    python -m src.preprocessing.preprocess --scans data/tables/pilot_scans.csv

The scan list needs an `image_id` column. Each scan is looked up under the raw scans folder
by its ADNI image folder (I<image_id>). Outputs, all named by image ID (never subject ID):
    data/processed/<image_id>.nii.gz          the preprocessed volume
    outputs/preprocessing/qc/<image_id>.png   three slices with the template brain outline
    outputs/preprocessing/qc_metrics.csv      one row per scan (status, time, QC score)
Scans already processed are skipped, so an interrupted run can simply be started again.
"""
import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from nilearn.datasets import fetch_icbm152_2009

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONFIG = Path("configs/preprocessing.yaml")


# ---------------------------------------------------------------- template
def load_template(template_dir: Path, shape: list):
    """Return the MNI152 head template and brain mask as SimpleITK images, plus the crop box.

    The ICBM152 2009 template keeps the skull, like our scans. A brain-only template makes
    the registration shrink the whole head to fit the brain outline.
    """
    files = fetch_icbm152_2009(data_dir=str(template_dir), verbose=0)
    template = sitk.DICOMOrient(sitk.ReadImage(files["t1"], sitk.sitkFloat32), "LPS")
    mask = sitk.DICOMOrient(sitk.ReadImage(files["mask"], sitk.sitkUInt8), "LPS")

    # Fixed-size box centred on the brain mask, kept inside the template grid
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
    start = []
    for lo, size, want, full in zip((x, y, z), (sx, sy, sz), shape, template.GetSize()):
        begin = lo + size // 2 - want // 2
        start.append(int(min(max(begin, 0), full - want)))
    return template, mask, start


# ---------------------------------------------------------------- steps
def read_dicom(folder: Path, orient: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(folder))
    if not files:
        raise RuntimeError("no DICOM series found")
    reader.SetFileNames(files)
    image = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
    return sitk.DICOMOrient(image, orient)


def n4_correct(image: sitk.Image, shrink: int) -> sitk.Image:
    """Remove slow brightness variation. The field is estimated on a shrunken copy."""
    head = sitk.OtsuThreshold(image, 0, 1, 200)
    small = sitk.Shrink(image, [shrink] * 3)
    small_head = sitk.Shrink(head, [shrink] * 3)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(small, small_head)
    field = corrector.GetLogBiasFieldAsImage(image)
    return sitk.Cast(image / sitk.Exp(field), sitk.sitkFloat32)


def head_top(image: sitk.Image, keep_mm: float = 180.0) -> sitk.Image:
    """The top part of the head, without the neck (in LPS orientation z increases upwards)."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.OtsuThreshold(image, 0, 1, 200))
    box = stats.GetBoundingBox(1)
    top = box[2] + box[5]
    bottom = max(top - int(keep_mm / image.GetSpacing()[2]), 0)
    size = list(image.GetSize())
    size[2] = top - bottom
    return sitk.RegionOfInterest(image, size, [0, 0, bottom])


def register(image: sitk.Image, template: sitk.Image, mask: sitk.Image) -> sitk.Transform:
    """Rigid then affine alignment to the template.

    Three choices made during the pilot (see docs/preprocessing_log.md):
    - The match is scored only inside the template's brain (mask shrunk by 3 mm). If the skull
      takes part, the bright scalp gets matched to the brain edge and the head is shrunk.
    - Similarity is plain correlation; mutual information failed on some pilot scans.
    - The starting position uses the top 180 mm of the head, so the neck does not pull the
      centre of the scan downwards.
    """
    region = sitk.BinaryErode(mask, [3, 3, 3])

    def run(transform, shrink, smooth):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsCorrelation()
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.25, 42)
        reg.SetMetricFixedMask(region)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0, minStep=0.005, numberOfIterations=300, relaxationFactor=0.6)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel(shrink)
        reg.SetSmoothingSigmasPerLevel(smooth)
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(transform, inPlace=True)
        reg.Execute(template, image)
        return transform

    rigid = sitk.CenteredTransformInitializer(
        template, head_top(image), sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS)
    rigid = run(rigid, [8, 4], [4, 2])
    affine = sitk.AffineTransform(3)
    affine.SetCenter(rigid.GetCenter())
    affine.SetMatrix(rigid.GetMatrix())
    affine.SetTranslation(rigid.GetTranslation())
    return run(affine, [4, 2], [2, 1])


def normalise(array: np.ndarray, brain: np.ndarray, clip: list) -> np.ndarray:
    """Clip extreme values, then z-score using voxels inside the template brain mask."""
    lo, hi = np.percentile(array[brain], clip)
    array = np.clip(array, lo, hi)
    return ((array - array[brain].mean()) / array[brain].std()).astype(np.float32)


def template_correlation(array: np.ndarray, template: np.ndarray, brain: np.ndarray) -> float:
    return float(np.corrcoef(array[brain], template[brain])[0, 1])


def save_qc_picture(array, brain, path: Path, title: str) -> None:
    """Middle slice in each direction, with the template brain outline in red."""
    mid = [s // 2 for s in array.shape]
    views = [(array[mid[0]], brain[mid[0]]), (array[:, mid[1]], brain[:, mid[1]]),
             (array[:, :, mid[2]], brain[:, :, mid[2]])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, (img, outline) in zip(axes, views):
        ax.imshow(img, cmap="gray", origin="lower", vmin=-2.5, vmax=2.5)
        ax.contour(outline, levels=[0.5], colors="red", linewidths=0.8)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- one scan
def process(image_id: int, cfg: dict, template, mask, start) -> dict:
    steps, out = cfg["steps"], cfg["output"]
    raw, out_dir, qc_dir = (Path(cfg["paths"][k]) for k in ("raw_scans", "out_dir", "qc_dir"))
    folders = list(raw.rglob(f"I{image_id}"))
    if not folders:
        raise FileNotFoundError("scan folder not found")

    image = read_dicom(folders[0], steps["orient"])
    in_size, in_spacing = image.GetSize(), [round(s, 2) for s in image.GetSpacing()]
    if steps["bias_correction"]:
        image = n4_correct(image, steps["n4_shrink_factor"])
    transform = register(image, template, mask)
    aligned = sitk.Resample(image, template, transform, sitk.sitkLinear, 0.0)

    crop = lambda img: sitk.RegionOfInterest(img, out["shape"], start)  # noqa: E731
    aligned, brain_img, tmpl_img = crop(aligned), crop(mask), crop(template)
    brain = sitk.GetArrayFromImage(brain_img) > 0
    array = normalise(sitk.GetArrayFromImage(aligned), brain, steps["clip_percentiles"])
    score = template_correlation(array, sitk.GetArrayFromImage(tmpl_img), brain)

    result = sitk.GetImageFromArray(array)
    result.CopyInformation(aligned)
    sitk.WriteImage(result, str(out_dir / f"{image_id}.nii.gz"))
    flagged = score < cfg["qc"]["min_template_correlation"]
    save_qc_picture(array, brain, qc_dir / "qc" / f"{image_id}.png",
                    f"image {image_id}   template correlation {score:.2f}"
                    + ("   FLAGGED" if flagged else ""))
    return {"input_size": "x".join(map(str, in_size)), "input_spacing": str(in_spacing),
            "template_correlation": round(score, 3), "flagged": flagged}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans", required=True, help="CSV with an image_id column")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    out_dir, qc_dir = Path(cfg["paths"]["out_dir"]), Path(cfg["paths"]["qc_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "qc").mkdir(parents=True, exist_ok=True)
    template, mask, start = load_template(Path(cfg["paths"]["template_dir"]),
                                          cfg["output"]["shape"])

    metrics_file = qc_dir / "qc_metrics.csv"
    rows = pd.read_csv(metrics_file).to_dict("records") if metrics_file.exists() else []
    done = {r["image_id"] for r in rows if r["status"] == "ok"}
    image_ids = pd.read_csv(args.scans)["image_id"].drop_duplicates().tolist()
    versions = f"SimpleITK {sitk.Version_VersionString()}"

    for number, image_id in enumerate(image_ids, 1):
        if image_id in done:
            continue
        began = time.time()
        row = {"image_id": image_id, "status": "ok", "error": "", "tool": versions}
        try:
            row.update(process(image_id, cfg, template, mask, start))
        except Exception as error:  # keep going; failures are logged for review
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        row["seconds"] = round(time.time() - began, 1)
        rows = [r for r in rows if r["image_id"] != image_id] + [row]
        pd.DataFrame(rows).to_csv(metrics_file, index=False)
        print(f"[{number}/{len(image_ids)}] {row['status']}  {row['seconds']}s  "
              f"correlation {row.get('template_correlation', '-')}")

    table = pd.DataFrame(rows)
    print("\nok:", int((table["status"] == "ok").sum()),
          "| failed:", int((table["status"] == "failed").sum()),
          "| flagged:", int(table.get("flagged", pd.Series(dtype=bool)).fillna(False).sum()))


if __name__ == "__main__":
    main()
