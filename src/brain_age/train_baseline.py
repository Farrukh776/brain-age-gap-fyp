"""Train the ridge / SVR brain-age baseline on 4 mm voxel features (decision 22).

Run from the repo root:  python -m src.brain_age.train_baseline
Quick test (small grid, 2 folds):  python -m src.brain_age.train_baseline --quick

Input: data/features/voxels_4mm.npy from src.brain_age.extract_features.
Pipeline per fit: per-voxel standardisation -> PCA -> ridge or SVR (RBF). Both the scaler and
the PCA are fitted on the training part of each fold only.

1. Tuning: 5-fold CV on CN training scans (all visits), folds grouped by subject, so all scans
   of one person are on the same side. For each fold, PCA is fitted once with the largest number
   of components and the first k columns are used for each smaller k (memory-friendly; the
   leading components do not depend on how many are kept).
2. For each model family, the setting with the lowest CV MAE is refitted on all training scans
   and evaluated on CN validation (one baseline scan per subject).
The test set is not loaded.

Writes (local only, outputs/brain_age/baseline/):
  cv_results.csv       mean / SD of CV MAE per setting (no subject rows)
  metrics.json         chosen settings, CV and validation metrics
  oof_train.csv        out-of-fold predictions for training scans (for bias correction later)
  val_predictions.csv  validation predictions
  <family>.joblib      fitted pipeline
Prints aggregate metrics only.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def metrics(age: np.ndarray, pred: np.ndarray) -> dict:
    gap = pred - age
    return {
        "n": int(len(age)),
        "mae": float(np.abs(gap).mean()),
        "rmse": float(np.sqrt((gap ** 2).mean())),
        "pearson_r": float(pearsonr(age, pred)[0]),
        "spearman_r": float(spearmanr(age, pred)[0]),
        "mean_gap": float(gap.mean()),
        # Regression to the mean: negative slope = young over-, old under-predicted
        "gap_vs_age_slope": float(np.polyfit(age, gap, 1)[0]),
    }


def make_model(family: str, params: dict):
    if family == "ridge":
        return Ridge(alpha=params["alpha"])
    return SVR(kernel="rbf", C=params["C"], epsilon=params["epsilon"], gamma="scale")


def settings(grid: dict) -> list:
    out = []
    for k in grid["pca_components"]:
        for a in grid["ridge"]["alpha"]:
            out.append(("ridge", k, {"alpha": a}))
        for c in grid["svr"]["C"]:
            for e in grid["svr"]["epsilon"]:
                out.append(("svr", k, {"C": c, "epsilon": e}))
    return out


def load_data(cfg: dict):
    feat_dir = Path(cfg["paths"]["features_dir"])
    X = np.load(feat_dir / "voxels_4mm.npy", mmap_mode="r")
    index = pd.read_csv(feat_dir / "voxels_4mm_index.csv")
    ages = pd.concat([
        pd.read_csv(cfg["paths"]["train_scans"])[["image_id", "age"]],
        pd.read_csv(cfg["paths"]["cohort_table"])[["image_id", "age"]],
    ]).drop_duplicates("image_id").set_index("image_id")["age"]
    index["age"] = index["image_id"].map(ages)
    cn = index["cohort"] == "CN"
    train = index[cn & (index["split"] == "train")]
    val = index[cn & (index["split"] == "val")]
    if train["age"].isna().any() or val["age"].isna().any():
        raise SystemExit("missing age for some CN scans")
    return X, train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/brain_age.yaml")
    parser.add_argument("--quick", action="store_true", help="small grid, 2 folds")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    bcfg = cfg["baseline"]
    grid = bcfg["grid"]
    folds = bcfg["cv_folds"]
    if args.quick:
        grid = {"pca_components": [50], "ridge": {"alpha": [1e3]},
                "svr": {"C": [10], "epsilon": [1.0]}}
        folds = 2
    seed = cfg["seed"]
    out = Path(cfg["paths"]["out_dir"]) / ("baseline_quick" if args.quick else "baseline")
    out.mkdir(parents=True, exist_ok=True)

    X, train, val = load_data(cfg)
    X_train = np.asarray(X[train.index.to_numpy()], dtype=np.float32)
    y_train = train["age"].to_numpy()
    groups = train["RID"].to_numpy()
    print(f"Training: {len(train)} scans of {train['RID'].nunique()} subjects; "
          f"validation: {len(val)} scans; {X_train.shape[1]} features")

    all_settings = settings(grid)
    k_max = max(grid["pca_components"])
    oof = {i: np.zeros(len(train)) for i in range(len(all_settings))}
    fold_mae = {i: [] for i in range(len(all_settings))}

    # GroupKFold with shuffle (sklearn >= 1.6) keeps folds subject-level and reproducible
    cv = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(cv.split(X_train, y_train, groups)):
        t0 = time.time()
        scaler = StandardScaler().fit(X_train[tr])
        pca = PCA(n_components=k_max, svd_solver="randomized", random_state=seed)
        z_tr = pca.fit_transform(scaler.transform(X_train[tr]))
        z_te = pca.transform(scaler.transform(X_train[te]))
        for i, (family, k, params) in enumerate(all_settings):
            pred = make_model(family, params).fit(z_tr[:, :k], y_train[tr]).predict(z_te[:, :k])
            oof[i][te] = pred
            fold_mae[i].append(float(np.abs(pred - y_train[te]).mean()))
        print(f"  fold {f + 1}/{folds}: {len(tr)} train / {len(te)} held-out scans, "
              f"{time.time() - t0:.0f} s")

    rows = [{"family": fam, "pca_components": k, **{p: params.get(p) for p in
             ("alpha", "C", "epsilon")}, "cv_mae": np.mean(fold_mae[i]),
             "cv_mae_sd": np.std(fold_mae[i])}
            for i, (fam, k, params) in enumerate(all_settings)]
    cv_table = pd.DataFrame(rows).sort_values("cv_mae")
    cv_table.to_csv(out / "cv_results.csv", index=False)

    results = {"seed": seed, "cv_folds": folds, "n_features": int(X_train.shape[1]),
               "n_train_scans": int(len(train)), "n_train_subjects": int(train["RID"].nunique())}
    X_val = np.asarray(X[val.index.to_numpy()], dtype=np.float32)
    y_val = val["age"].to_numpy()
    val_out = val[["image_id", "RID", "age"]].copy()
    oof_out = train[["image_id", "RID", "age", "is_baseline"]].copy()

    for family in ("ridge", "svr"):
        best = cv_table[cv_table["family"] == family].iloc[0]
        i = next(j for j, (fam, k, p) in enumerate(all_settings)
                 if fam == family and k == best["pca_components"]
                 and all(p[n] == best[n] for n in p))
        _, k, params = all_settings[i]
        model = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=k, svd_solver="randomized", random_state=seed)),
            ("model", make_model(family, params)),
        ]).fit(X_train, y_train)
        pred_val = model.predict(X_val)
        joblib.dump(model, out / f"{family}.joblib")
        val_out[f"pred_{family}"] = pred_val
        oof_out[f"oof_{family}"] = oof[i]
        results[family] = {
            "pca_components": int(k), "params": params,
            "cv": metrics(y_train, oof[i]),
            "cv_baseline_scans": metrics(y_train[train["is_baseline"].to_numpy()],
                                         oof[i][train["is_baseline"].to_numpy()]),
            "val": metrics(y_val, pred_val),
        }

    val_out.to_csv(out / "val_predictions.csv", index=False)
    oof_out.to_csv(out / "oof_train.csv", index=False)
    json.dump(results, open(out / "metrics.json", "w"), indent=2)

    print("\nBest setting per family (CV MAE on training scans, then validation):")
    for family in ("ridge", "svr"):
        r = results[family]
        print(f"  {family:5s} k={r['pca_components']:<4d} {r['params']}")
        for part in ("cv", "val"):
            m = r[part]
            print(f"     {part:3s}  n={m['n']:<5d} MAE {m['mae']:.2f}  RMSE {m['rmse']:.2f}  "
                  f"r {m['pearson_r']:.3f}  mean gap {m['mean_gap']:+.2f}  "
                  f"gap-vs-age slope {m['gap_vs_age_slope']:+.2f}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
