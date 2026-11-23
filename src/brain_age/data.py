"""Scan list and PyTorch dataset for the 3D CNN brain-age model.

Test it from the repo root:  python -m src.brain_age.data --n 20
(loads 20 training and some validation scans, checks shapes and values, times the loading,
and saves a picture of one sample with and without augmentation to outputs/brain_age/).

Scans are the same as for the baseline (verify_processed.load_scans): CN training = all
visits (decision 18), CN validation / test and MCI = one baseline scan per subject.
Each 1 mm scan (160 x 192 x 160) is shrunk to 2 mm by averaging 2 x 2 x 2 blocks, giving
80 x 96 x 80. Augmentation (training only): left-right flip and a random shift of up to
2 voxels per axis. Images are already z-scored inside the brain, so no further scaling.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.brain_age.extract_features import block_mean
from src.preprocessing.verify_processed import load_scans


def scan_table(cfg: dict) -> pd.DataFrame:
    """Every scan the models use, with age: cohort, split, RID, image_id, is_baseline, age."""
    scans = load_scans().drop(columns="age")
    ages = pd.concat([
        pd.read_csv(cfg["paths"]["train_scans"])[["image_id", "age"]],
        pd.read_csv(cfg["paths"]["cohort_table"])[["image_id", "age"]],
        pd.read_csv(cfg["paths"]["mci_cohort_table"])[["image_id", "age"]],
    ]).drop_duplicates("image_id").set_index("image_id")["age"]
    # A scan replaced by another T1 of the same session (decision 17) has a new image ID that
    # the cohort tables do not know; it has the age of the image it replaced
    qc = pd.read_csv(cfg["paths"]["scan_qc"]).dropna(subset=["image_id"])
    replaced = qc.set_index(qc["image_id"].astype("int64"))["original_image_id"]
    scans["age"] = scans["image_id"].map(ages).fillna(
        scans["image_id"].map(replaced).map(ages))
    if scans["age"].isna().any():
        raise SystemExit("some scans have no age")
    return scans


class BrainAgeDataset(Dataset):
    """Returns (image [1, 80, 96, 80] float32, age float32, row number in the table)."""

    def __init__(self, table: pd.DataFrame, processed_dir: str, factor: int = 2,
                 augment: bool = False, max_shift: int = 2):
        self.table = table.reset_index(drop=True)
        self.dir = Path(processed_dir)
        self.factor = factor
        self.augment = augment
        self.max_shift = max_shift

    def __len__(self) -> int:
        return len(self.table)

    def load(self, i: int) -> np.ndarray:
        path = self.dir / f"{self.table.at[i, 'image_id']}.nii.gz"
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(path), sitk.sitkFloat32))  # z, y, x
        return block_mean(array, self.factor).astype(np.float32)

    def augment_image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < 0.5:
            image = image[:, :, ::-1]  # x axis = left-right (scans are in LPS orientation)
        s = self.max_shift
        if s:
            padded = np.pad(image, s, mode="constant", constant_values=float(image.min()))
            dz, dy, dx = rng.integers(0, 2 * s + 1, size=3)
            z, y, x = image.shape
            image = padded[dz:dz + z, dy:dy + y, dx:dx + x]
        return np.ascontiguousarray(image)

    def __getitem__(self, i: int):
        image = self.load(i)
        if self.augment:
            # torch seeds each worker per epoch; draw from it so runs are reproducible
            rng = np.random.default_rng(torch.randint(0, 2**31, (1,)).item())
            image = self.augment_image(image, rng)
        age = np.float32(self.table.at[i, "age"])
        return torch.from_numpy(image[None]), torch.tensor(age), i


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int,
                seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=g, persistent_workers=workers > 0)


def save_picture(dataset: BrainAgeDataset, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plain = dataset.load(0)
    aug = dataset.augment_image(plain, np.random.default_rng(1))
    flipped = plain[:, :, ::-1]
    rows = [("no augmentation", plain), ("left-right flip", flipped),
            ("augmented (random)", aug)]
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for r, (name, img) in enumerate(rows):
        z, y, x = (s // 2 for s in img.shape)
        # SimpleITK arrays from LPS images: z = inferior->superior, y = anterior->posterior
        views = [img[z], img[:, y][::-1], img[:, :, x][::-1]]
        for c, (view, title) in enumerate(zip(views, ["axial", "coronal", "sagittal"])):
            axes[r, c].imshow(view, cmap="gray", vmin=-2, vmax=3)
            axes[r, c].set_title(f"{name}: {title}", fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle("3D CNN input at 2 mm (80 x 96 x 80)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the 3D CNN data loader")
    parser.add_argument("--config", default="configs/brain_age.yaml")
    parser.add_argument("--n", type=int, default=20, help="training scans to load")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    prep = yaml.safe_load(open(cfg["paths"]["preprocessing_config"]))
    ccfg = cfg["cnn"]
    factor = int(ccfg["input_downsample_mm"])
    seed = cfg["seed"]
    torch.manual_seed(seed)

    table = scan_table(cfg)
    counts = table.groupby(["cohort", "split"]).size()
    print("Scan table (all scans):\n" + counts.to_string())
    cn = table[table["cohort"] == "CN"]
    train = cn[cn["split"] == "train"].sample(args.n, random_state=seed)
    val = cn[cn["split"] == "val"].head(max(args.n // 4, 4))

    problems = []
    if set(train["RID"]) & set(val["RID"]):
        problems.append("a subject is in both train and validation")

    ds_train = BrainAgeDataset(train, prep["paths"]["out_dir"], factor, augment=True,
                               max_shift=ccfg["augment_max_shift"])
    ds_val = BrainAgeDataset(val, prep["paths"]["out_dir"], factor, augment=False)

    t0 = time.time()
    loader = make_loader(ds_train, ccfg["batch_size"], True, args.workers, seed)
    seen, n_batches = [], 0
    for images, ages, idx in loader:
        n_batches += 1
        seen += idx.tolist()
        if images.shape[1:] != (1, 80, 96, 80) or images.dtype != torch.float32:
            problems.append(f"bad batch shape or type {tuple(images.shape)} {images.dtype}")
        if not torch.isfinite(images).all():
            problems.append("NaN or inf in a batch")
        if ages.min() < 40 or ages.max() > 100:
            problems.append("age outside 40-100")
    t_train = time.time() - t0
    if sorted(seen) != list(range(len(ds_train))):
        problems.append("loader did not return every training scan exactly once")

    # Validation must be identical every time (no augmentation)
    a, b = ds_val[0][0], ds_val[0][0]
    if not torch.equal(a, b):
        problems.append("validation images change between reads")
    # Augmentation must change training images but keep their values
    img = ds_train.load(0)
    aug = ds_train.augment_image(img, np.random.default_rng(3))
    if aug.shape != img.shape:
        problems.append("augmentation changed the image shape")

    vals = torch.stack([ds_val[i][0] for i in range(len(ds_val))])
    out = Path(cfg["paths"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_picture(ds_val, out / "loader_check.png")

    print(f"\nTrain: {len(ds_train)} scans in {n_batches} batches of up to "
          f"{ccfg['batch_size']}, {t_train:.1f} s ({t_train / len(ds_train):.2f} s per scan, "
          f"{args.workers} workers)")
    print(f"Validation: {len(ds_val)} scans; image mean {vals.mean():.3f}, SD {vals.std():.3f}, "
          f"min {vals.min():.2f}, max {vals.max():.2f}")
    print(f"Memory per scan: {vals[0].numel() * 4 / 1e6:.1f} MB; picture: {out / 'loader_check.png'}")
    print("\nPROBLEMS:\n  " + "\n  ".join(problems) if problems else "\nAll loader checks passed.")


if __name__ == "__main__":
    main()
