"""Fit the linear age-bias correction on training data only and check it on validation (decision 30).

    python -m src.brain_age.bias_correction [--cv-run cnn-cv]

For each model, on the out-of-fold predictions of the 1,755 CN training scans:
    gap = predicted - true age;   fit  gap = a * age + b   (least squares)
For any scan afterwards:
    corrected BAG = gap - (a * age + b)
Models:
    cnn    average of the 5 fold CNNs (cnn-cv/fold0..4); fitted on their out-of-fold predictions
    ridge  ridge baseline (decision 24); fitted on its out-of-fold predictions (oof_train.csv)
Validation (140 CN baselines) is only used to check the result: after correction the gap should
no longer depend on age (slope and correlation near 0) and average about 0. MAE is always
the uncorrected one: the corrected gap uses the true age, so a "corrected MAE" would flatter.

Writes (local only) outputs/brain_age/bias/: correction.json (a, b per model; no subject data),
val_corrected.csv, age_bias_corrected.png. Prints summary numbers only. Test set not read.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.brain_age.train_baseline import metrics  # noqa: E402

N_FOLDS = 5


def fit(age: np.ndarray, pred: np.ndarray) -> dict:
    a, b = np.polyfit(age, pred - age, 1)
    return {"a": float(a), "b": float(b)}


def corrected_gap(age, pred, params: dict) -> np.ndarray:
    return (np.asarray(pred) - np.asarray(age)) - (params["a"] * np.asarray(age) + params["b"])


def gap_summary(age: np.ndarray, gap: np.ndarray) -> dict:
    return {"mean": float(gap.mean()), "sd": float(gap.std(ddof=1)),
            "slope_vs_age": float(np.polyfit(age, gap, 1)[0]),
            "r_with_age": float(np.corrcoef(age, gap)[0, 1])}


def load(out: Path, cv_run: str) -> dict:
    """Training out-of-fold and validation predictions per model, aligned by image_id."""
    cv = out / "cnn" / cv_run
    oof = pd.concat([pd.read_csv(cv / f"fold{k}" / "oof_predictions.csv") for k in range(N_FOLDS)])
    if oof["image_id"].duplicated().any():
        raise SystemExit("a training scan has more than one out-of-fold prediction")
    val = [pd.read_csv(cv / f"fold{k}" / "val_predictions.csv").set_index("image_id")
           for k in range(N_FOLDS)]
    val_cnn = val[0][["RID", "age"]].assign(
        pred=pd.concat([v["pred_cnn"] for v in val], axis=1).mean(axis=1))

    r_oof = pd.read_csv(out / "baseline" / "oof_train.csv")
    r_val = pd.read_csv(out / "baseline" / "val_predictions.csv").set_index("image_id")
    if set(oof["image_id"]) != set(r_oof["image_id"]) or set(val_cnn.index) != set(r_val.index):
        raise SystemExit("CNN and ridge predictions cover different scans")
    return {
        "cnn": (oof["age"].to_numpy(), oof["pred_cnn"].to_numpy(),
                val_cnn["age"].to_numpy(), val_cnn["pred"].to_numpy(), val_cnn.index),
        "ridge": (r_oof["age"].to_numpy(), r_oof["oof_ridge"].to_numpy(),
                  r_val["age"].to_numpy(), r_val["pred_ridge"].to_numpy(), r_val.index),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/brain_age.yaml")
    parser.add_argument("--cv-run", default="cnn-cv")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["paths"]["out_dir"])
    dest = out / "bias"
    dest.mkdir(parents=True, exist_ok=True)

    data = load(out, args.cv_run)
    result, rows = {}, []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    for j, (name, (age_tr, pred_tr, age_v, pred_v, ids)) in enumerate(data.items()):
        params = fit(age_tr, pred_tr)
        raw_v = pred_v - age_v
        cor_v = corrected_gap(age_v, pred_v, params)
        cor_tr = corrected_gap(age_tr, pred_tr, params)
        result[name] = {
            "fitted_on": f"out-of-fold predictions, {len(age_tr)} CN training scans",
            "params": params,
            "train_oof_accuracy": metrics(age_tr, pred_tr),
            "val_accuracy_uncorrected": metrics(age_v, pred_v),
            "val_gap_before": gap_summary(age_v, raw_v),
            "val_gap_after": gap_summary(age_v, cor_v),
            "train_gap_after": gap_summary(age_tr, cor_tr),
        }
        rows.append(pd.DataFrame({"image_id": ids, f"pred_{name}": pred_v,
                                  f"bag_raw_{name}": raw_v, f"bag_{name}": cor_v}))
        for i, (gap, label) in enumerate([(raw_v, "before"), (cor_v, "after")]):
            ax = axes[i, j]
            s = gap_summary(age_v, gap)
            xs = np.array([50, 95])
            k, c = np.polyfit(age_v, gap, 1)
            ax.scatter(age_v, gap, s=8, alpha=0.5)
            ax.axhline(0, color="k", ls="--", lw=1)
            ax.plot(xs, k * xs + c, "r", lw=2)
            ax.set(title=f"{name.upper()} validation, {label} correction\n"
                         f"slope {s['slope_vs_age']:+.2f}, r with age {s['r_with_age']:+.2f}, "
                         f"mean {s['mean']:+.2f}",
                   xlim=(50, 95), ylim=(-20, 20))
            if i == 1:
                ax.set_xlabel("true age")
            if j == 0:
                ax.set_ylabel("BAG (years)")
    fig.suptitle("Age-bias correction fitted on training (out-of-fold) only; checked on validation")
    fig.tight_layout()
    fig.savefig(dest / "age_bias_corrected.png", dpi=110)

    val_out = rows[0].merge(rows[1], on="image_id")
    val_out.to_csv(dest / "val_corrected.csv", index=False)
    json.dump(result, open(dest / "correction.json", "w"), indent=2)

    print(f"{'model':6s} {'a':>7s} {'b':>7s} | {'val MAE':>7s} {'r':>6s} | val gap slope / r with age / mean / SD")
    for name, r in result.items():
        bef, aft = r["val_gap_before"], r["val_gap_after"]
        print(f"{name:6s} {r['params']['a']:+7.3f} {r['params']['b']:+7.2f} | "
              f"{r['val_accuracy_uncorrected']['mae']:7.2f} "
              f"{r['val_accuracy_uncorrected']['pearson_r']:6.3f} | "
              f"before {bef['slope_vs_age']:+.3f} / {bef['r_with_age']:+.2f} / {bef['mean']:+.2f} / "
              f"{bef['sd']:.2f}   after {aft['slope_vs_age']:+.3f} / {aft['r_with_age']:+.2f} / "
              f"{aft['mean']:+.2f} / {aft['sd']:.2f}")
    print(f"Saved {dest / 'correction.json'}, val_corrected.csv, age_bias_corrected.png")


if __name__ == "__main__":
    main()
