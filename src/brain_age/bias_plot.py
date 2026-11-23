"""Age-bias plot for the brain-age models (step 6), before any correction is chosen.

Brain-age models pull predictions towards the mean training age ("regression to the mean"):
young people get too old a brain age, old people too young. Then the gap (predicted - true age)
depends on age, and a BAG would partly measure age itself. This script shows how strong that
effect is for each model and data set available now:
    ridge  out-of-fold training predictions (oof_train.csv, 1,755 scans; honest, never seen)
    ridge  validation (140)
    CNN    validation (140)  (no honest CNN training predictions exist yet)

    python -m src.brain_age.bias_plot [--cnn-run cnn-1]

Writes outputs/brain_age/bias/age_bias.png (local only: one point per scan) and prints slopes
and summary metrics only. The test set is not read.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.brain_age.train_baseline import metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/brain_age.yaml")
    parser.add_argument("--cnn-run", default="cnn-1")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["paths"]["out_dir"])

    oof = pd.read_csv(out / "baseline" / "oof_train.csv")
    val_r = pd.read_csv(out / "baseline" / "val_predictions.csv")
    val_c = pd.read_csv(out / "cnn" / args.cnn_run / "val_predictions.csv")
    sets = [("Ridge, training (out-of-fold)", oof["age"].values, oof["oof_ridge"].values),
            ("Ridge, validation", val_r["age"].values, val_r["pred_ridge"].values),
            ("3D CNN, validation", val_c["age"].values, val_c["pred_cnn"].values)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    lo, hi = 50, 95
    print(f"{'set':32s} {'n':>5s} {'MAE':>5s} {'r':>6s}  {'gap slope':>9s}  {'pred slope':>10s}")
    for j, (name, age, pred) in enumerate(sets):
        m = metrics(age, pred)
        a_pred, b_pred = np.polyfit(age, pred, 1)
        a_gap, b_gap = np.polyfit(age, pred - age, 1)
        print(f"{name:32s} {m['n']:5d} {m['mae']:5.2f} {m['pearson_r']:6.3f}  {a_gap:9.3f}  "
              f"{a_pred:10.3f}")
        xs = np.array([lo, hi])

        ax = axes[0, j]
        ax.scatter(age, pred, s=6, alpha=0.4)
        ax.plot(xs, xs, "k--", lw=1, label="perfect (predicted = true)")
        ax.plot(xs, a_pred * xs + b_pred, "r", lw=2, label=f"fit, slope {a_pred:.2f}")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="true age", ylabel="predicted age",
               title=f"{name}\nMAE {m['mae']:.2f}, r {m['pearson_r']:.2f}")
        ax.legend(loc="upper left", fontsize=8)

        ax = axes[1, j]
        ax.scatter(age, pred - age, s=6, alpha=0.4)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.plot(xs, a_gap * xs + b_gap, "r", lw=2, label=f"fit, slope {a_gap:.2f}")
        ax.set(xlim=(lo, hi), ylim=(-20, 20), xlabel="true age",
               ylabel="gap = predicted - true age")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Age bias before correction (CN subjects; test set not used)")
    fig.tight_layout()
    path = out / "bias" / "age_bias.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
