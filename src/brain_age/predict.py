"""Predict brain age and corrected BAG with the frozen final model (decision 31).

    python -m src.brain_age.predict --set cn_val                    # check: must match validation
    python -m src.brain_age.predict --set cn_test --confirm-test    # ONCE, after freezing
    python -m src.brain_age.predict --set mci                       # step 7 (BAG for MCI)

Final model: average of the 5 fold CNNs (configs `final`); weights are checked against the
SHA-256 values in the config before use. Corrected BAG = (predicted - age) - (a * age + b) with
the training-fitted correction (correction.json, key "cnn").
CN sets also print uncorrected accuracy (MAE, RMSE, r with bootstrap 95% CI for MAE) and how
much the corrected BAG still depends on age. For cn_test, cnn-1 and the ridge baseline are
reported alongside for comparison only (decision 31). The test set runs once: the script
refuses if test predictions already exist.

Writes (local only) outputs/brain_age/final/<set>_predictions.csv (+ <set>_report.json).
Prints summary numbers only. CPU is enough (about 0.3 s per scan).
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.brain_age.bias_correction import corrected_gap, gap_summary
from src.brain_age.cnn import BrainAgeCNN
from src.brain_age.data import BrainAgeDataset, make_loader, scan_table
from src.brain_age.train_baseline import metrics
from src.brain_age.train_cnn import predict

SETS = {"cn_val": ("CN", "val"), "cn_test": ("CN", "test"), "mci": ("MCI", None)}


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def cnn_predict(run_dir: Path, ds, device, workers: int, ccfg: dict) -> np.ndarray:
    info = json.load(open(run_dir / "metrics.json"))
    model = BrainAgeCNN(dropout=ccfg["dropout"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    loader = make_loader(ds, 8, False, workers, 42)
    return predict(model, loader, device, info["target_mean"], info["target_sd"])


def mae_ci(age: np.ndarray, pred: np.ndarray, n_boot: int = 5000) -> list:
    err = np.abs(pred - age)
    idx = np.random.default_rng(42).integers(0, len(err), (n_boot, len(err)))
    return [float(x) for x in np.percentile(err[idx].mean(axis=1), [2.5, 97.5])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/brain_age.yaml")
    parser.add_argument("--set", required=True, choices=list(SETS))
    parser.add_argument("--confirm-test", action="store_true",
                        help="required for cn_test: the model is frozen (decision 31)")
    parser.add_argument("--cache-dir", help="2 mm cache (optional; same arrays)")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    fcfg, ccfg = cfg["final"], cfg["cnn"]
    prep = yaml.safe_load(open(cfg["paths"]["preprocessing_config"]))
    out_root = Path(cfg["paths"]["out_dir"])
    dest = out_root / "final"
    dest.mkdir(parents=True, exist_ok=True)
    out_csv = dest / f"{args.set}_predictions.csv"
    if args.set == "cn_test":
        if not args.confirm_test:
            raise SystemExit("cn_test needs --confirm-test (run once, after the model is frozen)")
        if out_csv.exists():
            raise SystemExit(f"{out_csv} exists: the test set has already been used once")

    # Frozen weights and correction must be exactly the ones recorded in decision 31
    cv_dir = out_root / "cnn" / fcfg["cv_run"]
    fold_dirs = [cv_dir / f"fold{k}" for k in range(fcfg["folds"])]
    found = [sha12(d / "best.pt") for d in fold_dirs] + [sha12(Path(fcfg["correction"]))]
    expected = fcfg["weights_sha256_12"] + [fcfg["correction_sha256_12"]]
    if found != expected:
        raise SystemExit(f"frozen files changed: {found} != {expected}")
    params = json.load(open(fcfg["correction"]))["cnn"]["params"]

    cohort, split = SETS[args.set]
    table = scan_table(cfg)
    rows = table[table["cohort"] == cohort]
    if split:
        rows = rows[rows["split"] == split]
    rows = rows.reset_index(drop=True)
    if rows["age"].isna().any():
        raise SystemExit("some scans have no age")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = BrainAgeDataset(rows, prep["paths"]["out_dir"], int(ccfg["input_downsample_mm"]),
                         augment=False, cache_dir=args.cache_dir or fcfg.get("cache_dir"))
    print(f"{args.set}: {len(rows)} scans of {rows['RID'].nunique()} subjects; device {device}")

    folds = np.stack([cnn_predict(d, ds, device, args.workers, ccfg) for d in fold_dirs])
    pred = folds.mean(axis=0)
    age = rows["age"].to_numpy()
    res = rows[["image_id", "RID", "cohort", "split", "age"]].copy()
    for k in range(len(fold_dirs)):
        res[f"pred_fold{k}"] = folds[k]
    res["brain_age"] = pred
    res["bag_raw"] = pred - age
    res["bag"] = corrected_gap(age, pred, params)

    report = {"set": args.set, "n_scans": len(rows), "n_subjects": int(rows["RID"].nunique()),
              "model": "cnn_ensemble (decision 31)", "correction": params,
              "bag_corrected": gap_summary(age, res["bag"].to_numpy()),
              "bag_raw": gap_summary(age, res["bag_raw"].to_numpy())}
    if cohort == "CN":
        report["accuracy_uncorrected"] = {"final": metrics(age, pred) | {"mae_ci95": mae_ci(age, pred)}}
        if args.set == "cn_test":  # comparison only (decision 31)
            p1 = cnn_predict(out_root / "cnn" / "cnn-1", ds, device, args.workers, ccfg)
            feat = Path(cfg["paths"]["features_dir"])
            index = pd.read_csv(feat / "voxels_4mm_index.csv")
            pos = index.reset_index().set_index("image_id").loc[rows["image_id"], "index"]
            X = np.load(feat / "voxels_4mm.npy", mmap_mode="r")[pos.to_numpy()]
            pr = joblib.load(out_root / "baseline" / "ridge.joblib").predict(
                np.asarray(X, dtype=np.float32))
            res["pred_cnn1"], res["pred_ridge"] = p1, pr
            for name, p in [("cnn-1", p1), ("ridge", pr)]:
                report["accuracy_uncorrected"][name] = metrics(age, p) | {"mae_ci95": mae_ci(age, p)}
            d = np.abs(pr - age) - np.abs(pred - age)
            idx = np.random.default_rng(42).integers(0, len(d), (5000, len(d)))
            report["final_minus_ridge_mae"] = {
                "diff": float(-d.mean()),
                "ci95": [float(-x) for x in np.percentile(d[idx].mean(axis=1), [97.5, 2.5])]}

    res.to_csv(out_csv, index=False)
    json.dump(report, open(dest / f"{args.set}_report.json", "w"), indent=2)

    if cohort == "CN":
        print(f"{'model':8s} {'MAE':>5s} {'95% CI':>13s} {'RMSE':>5s} {'r':>6s} {'mean gap':>8s} "
              f"{'gap slope':>9s}")
        for name, m in report["accuracy_uncorrected"].items():
            print(f"{name:8s} {m['mae']:5.2f} [{m['mae_ci95'][0]:.2f}, {m['mae_ci95'][1]:.2f}] "
                  f"{m['rmse']:5.2f} {m['pearson_r']:6.3f} {m['mean_gap']:+8.2f} "
                  f"{m['gap_vs_age_slope']:+9.3f}")
        if "final_minus_ridge_mae" in report:
            f = report["final_minus_ridge_mae"]
            print(f"final - ridge MAE {f['diff']:+.2f} (95% CI {f['ci95'][0]:+.2f} to {f['ci95'][1]:+.2f})")
    b, r = report["bag_corrected"], report["bag_raw"]
    print(f"BAG raw:       mean {r['mean']:+.2f}  SD {r['sd']:.2f}  slope vs age {r['slope_vs_age']:+.3f}"
          f"  r with age {r['r_with_age']:+.2f}")
    print(f"BAG corrected: mean {b['mean']:+.2f}  SD {b['sd']:.2f}  slope vs age {b['slope_vs_age']:+.3f}"
          f"  r with age {b['r_with_age']:+.2f}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
