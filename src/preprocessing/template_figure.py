"""Report figure of the registration template (MNI ICBM152 2009a, nonlinear symmetric, 1 mm).

Run from the repo root:  python -m src.preprocessing.template_figure

Draws the template T1 in three views with its brain mask (red; registration is scored inside
it, and intensities are z-scored inside it) and the 160 x 192 x 160 mm box every processed scan
is cropped to (yellow). Writes docs/figures/mni_template.png (public template, no ADNI data).
Template: Fonov et al. 2009/2011, McConnell Brain Imaging Centre, MNI, McGill University;
downloaded by nilearn (fetch_icbm152_2009); copyright notice in data/raw/templates/.
"""
from pathlib import Path

import matplotlib
import numpy as np
import SimpleITK as sitk
import yaml

from src.preprocessing.preprocess import load_template

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = Path("docs/figures/mni_template.png")


def main() -> None:
    cfg = yaml.safe_load(open("configs/preprocessing.yaml"))
    shape = cfg["output"]["shape"]
    template, mask, start = load_template(Path(cfg["paths"]["template_dir"]), shape)
    t = sitk.GetArrayFromImage(template)          # axes: z (inf->sup), y, x
    m = sitk.GetArrayFromImage(mask) > 0
    (x0, y0, z0), (sx, sy, sz) = start, shape
    cx, cy, cz = x0 + sx // 2, y0 + sy // 2, z0 + sz // 2

    # LPS orientation: flip so that anterior is to the right in the sagittal view and the
    # patient's right is on the image left (radiological convention) in coronal / axial views.
    views = [("Sagittal", t[:, :, cx], m[:, :, cx], (y0, z0, sy, sz), True),
             ("Coronal", t[:, cy, :], m[:, cy, :], (x0, z0, sx, sz), False),
             ("Axial", t[cz], m[cz], (x0, y0, sx, sy), False)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, (name, img, msk, (a, b, w, h), flip) in zip(axes, views):
        if flip:
            img, msk = img[:, ::-1], msk[:, ::-1]
            a = img.shape[1] - a - w
        ax.imshow(img, cmap="gray", origin="lower")
        ax.contour(msk, levels=[0.5], colors="red", linewidths=1)
        ax.add_patch(Rectangle((a - 0.5, b - 0.5), w, h, fill=False, ec="gold", lw=1.5))
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Registration template: MNI ICBM152 2009a nonlinear symmetric, 1 mm, with skull")
    fig.text(0.5, 0.02, "Red: template brain mask (registration scoring and intensity "
             "normalisation).  Yellow: 160 x 192 x 160 mm output box.\n"
             "Template (c) 1993-2004 Louis Collins, McConnell Brain Imaging Centre, Montreal "
             "Neurological Institute, McGill University.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("wrote", OUT, "| voxels:", "x".join(map(str, template.GetSize())),
          "| box start (x, y, z):", start, "| brain voxels:", int(m.sum()))


if __name__ == "__main__":
    main()
