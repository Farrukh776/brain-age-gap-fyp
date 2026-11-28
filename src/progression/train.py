"""Train and validate the MCI progression models for one experiment (E1, E2; decision 33).

Run from the repo root:
    python -m src.progression.train --config configs/e1.yaml
    python -m src.progression.train --config configs/e2.yaml

The same code runs every experiment; the config decides the inputs (E1: MRI features only,
E2: plus age). Steps:
  1. Load the MCI voxel features (src/features/mci_features.py) and the labels and age from
     mci_cohort.csv. Test subjects are dropped straight away: this script never reads them.
  2. Stratified 5-fold CV on MCI train (seed 42; same folds in every experiment, checked by a
     printed fingerprint). In each fold, fitted on the training part only:
     per-voxel standardisation -> PCA -> standardised PCA scores (+ standardised age in E2)
     -> L2 logistic regression or random forest (balanced class weights).
  3. For each model, the grid point with the highest mean CV ROC-AUC is chosen. Its out-of-fold
     training predictions give the probability cut-off (maximum Youden's J).
  4. The chosen setting is refitted on all of MCI train and scored once on MCI validation:
     ROC-AUC and PR-AUC (bootstrap 95% CI), sensitivity, specificity, precision, F1, confusion
     matrix, all at the training-derived cut-off.

Writes to <out_dir> (local only; ADNI-derived): cv_results.csv, oof_train.csv,
val_predictions.csv, metrics.json, models/<model>.joblib. Prints summary numbers only.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------- data

def load_data(cfg: dict):
    paths = cfg["paths"]
    index = pd.read_csv(paths["features_index"])
    X = np.load(paths["features"], mmap_mode="r")
    if len(index) != X.shape[0]:
        raise SystemExit("feature matrix and its index have different row counts")
    cohort = pd.read_csv(paths["cohort_table"])[["RID", "age", "label"]]
    t = index.reset_index().merge(cohort, on="RID", how="left", validate="one_to_one")
    if t[["age", "label"]].isna().any().any():
        raise SystemExit("some subjects have no age or label")
    splits = pd.read_csv(paths["splits"]).rename(columns={"split": "split_table"})
    t = t.merge(splits, on="RID", how="left", validate="one_to_one")
    if (t["split"] != t["split_table"]).any():
        raise SystemExit("split differs from mci_splits.csv")

    t = t[t["split"].isin(["train", "val"])]           # MCI test is never read here
    data = {}
    for split in ("train", "val"):
        s = t[t["split"] == split].reset_index(drop=True)
        data[split] = dict(
            rid=s["RID"].to_numpy(),
            X=np.asarray(X[s["index"].to_numpy()], dtype=np.float32),
            age=s["age"].to_numpy(dtype=float).reshape(-1, 1),
            y=s["label"].to_numpy(dtype=int),
        )
    return data


# ---------------------------------------------------------------- features

class Features:
    """Standardise voxels -> PCA (k_max) -> standardise the first k scores [+ standardised age].

    Fitted on training rows only. PCA is fitted once with k_max components and the first k are
    used, so every k in the grid shares one decomposition (as in the brain-age baseline)."""

    def __init__(self, k_max: int, use_age: bool, seed: int):
        self.k_max, self.use_age, self.seed = k_max, use_age, seed

    def fit(self, X, age):
        self.vox = StandardScaler().fit(X)
        self.pca = PCA(n_components=self.k_max, svd_solver="randomized",
                       random_state=self.seed).fit(self.vox.transform(X))
        scores = self.pca.transform(self.vox.transform(X))
        self.score_scale = StandardScaler().fit(scores)
        self.age_scale = StandardScaler().fit(age)
        return self

    def transform(self, X, age, k: int):
        scores = self.score_scale.transform(self.pca.transform(self.vox.transform(X)))[:, :k]
        if self.use_age:
            return np.hstack([scores, self.age_scale.transform(age)])
        return scores


def grid_points(cfg: dict, model: str):
    g = cfg["grid"]
    for k in g["pca_components"]:
        if model == "logistic_regression":
            for c in g["logistic_regression"]["C"]:
                yield dict(k=k, C=c)
        else:
            rf = g["random_forest"]
            for leaf in rf["min_samples_leaf"]:
                for mf in rf["max_features"]:
                    yield dict(k=k, min_samples_leaf=leaf, max_features=mf)


def make_classifier(cfg: dict, model: str, p: dict, seed: int):
    if model == "logistic_regression":
        return LogisticRegression(C=p["C"], class_weight="balanced", max_iter=10000)
    return RandomForestClassifier(
        n_estimators=cfg["grid"]["random_forest"]["n_estimators"],
        min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
        class_weight="balanced", random_state=seed, n_jobs=-1)


# ---------------------------------------------------------------- metrics

def youden_cutoff(y, prob) -> float:
    fpr, tpr, thr = roc_curve(y, prob)
    best = np.argmax(tpr - fpr)
    return float(min(thr[best], 1.0))


def threshold_metrics(y, prob, cutoff: float) -> dict:
    pred = (prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                precision=precision_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))


def bootstrap_ci(y, prob, fn, n: int, seed: int):
    rng = np.random.default_rng(seed)
    vals = []
    while len(vals) < n:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():                     # both classes needed
            vals.append(fn(y[i], prob[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed, exp = cfg["seed"], cfg["experiment"]
    inputs = cfg["inputs"]
    if inputs["bag"] or inputs["clinical"] or not inputs["mri_features"]:
        raise SystemExit("this script supports MRI features with or without age only (FYP-I)")
    use_age = bool(inputs["age"])
    out = Path(cfg["paths"]["out_dir"])
    (out / "models").mkdir(parents=True, exist_ok=True)

    data = load_data(cfg)
    tr, va = data["train"], data["val"]
    print(f"{exp}: inputs = MRI voxels{' + age' if use_age else ''}; train {len(tr['y'])} "
          f"({tr['y'].sum()} progressors), val {len(va['y'])} ({va['y'].sum()} progressors); "
          f"{tr['X'].shape[1]} voxel features. Test not loaded.")

    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)
    folds = list(cv.split(tr["X"], tr["y"]))
    fold_of = np.empty(len(tr["y"]), dtype=int)
    for f, (_, held) in enumerate(folds):
        fold_of[held] = f
    order = np.argsort(tr["rid"])
    fingerprint = hashlib.sha256(
        ",".join(f"{r}:{f}" for r, f in zip(tr["rid"][order], fold_of[order])).encode()
    ).hexdigest()[:12]
    print(f"CV folds: {cfg['cv_folds']} stratified, fingerprint {fingerprint} "
          f"(must be the same in every experiment)")

    k_max = max(cfg["grid"]["pca_components"])
    models = cfg["models"]
    # oof[model][grid index] = out-of-fold probabilities on train
    rows, oof = [], {m: {} for m in models}
    t0 = time.time()
    for f, (fit_i, held_i) in enumerate(folds):
        feats = Features(k_max, use_age, seed).fit(tr["X"][fit_i], tr["age"][fit_i])
        for m in models:
            for gi, p in enumerate(grid_points(cfg, m)):
                Z_fit = feats.transform(tr["X"][fit_i], tr["age"][fit_i], p["k"])
                Z_held = feats.transform(tr["X"][held_i], tr["age"][held_i], p["k"])
                clf = make_classifier(cfg, m, p, seed).fit(Z_fit, tr["y"][fit_i])
                prob = clf.predict_proba(Z_held)[:, 1]
                oof[m].setdefault(gi, np.full(len(tr["y"]), np.nan))[held_i] = prob
                rows.append(dict(model=m, grid=gi, fold=f, **p,
                                 auc=roc_auc_score(tr["y"][held_i], prob)))
        print(f"  fold {f} done ({time.time() - t0:.0f} s)")

    cv_table = pd.DataFrame(rows)
    keys = [c for c in cv_table.columns if c not in ("fold", "auc")]
    summary = (cv_table.groupby(keys, dropna=False)["auc"].agg(["mean", "std"]).reset_index()
               .rename(columns={"mean": "cv_auc_mean", "std": "cv_auc_sd"}))
    summary.to_csv(out / "cv_results.csv", index=False)

    metrics = dict(experiment=exp, config=args.config, git_commit=git_commit(),
                   fold_fingerprint=fingerprint, n_train=int(len(tr["y"])),
                   n_val=int(len(va["y"])), models={})
    oof_out = pd.DataFrame({"RID": tr["rid"], "label": tr["y"], "fold": fold_of})
    val_out = pd.DataFrame({"RID": va["rid"], "label": va["y"]})
    feats_all = Features(k_max, use_age, seed).fit(tr["X"], tr["age"])
    n_boot = cfg["bootstrap_iterations"]
    for m in models:
        s = summary[summary["model"] == m]
        best = s.loc[s["cv_auc_mean"].idxmax()]
        gi = int(best["grid"])
        p = list(grid_points(cfg, m))[gi]
        prob_oof = oof[m][gi]
        cutoff = youden_cutoff(tr["y"], prob_oof)

        clf = make_classifier(cfg, m, p, seed).fit(
            feats_all.transform(tr["X"], tr["age"], p["k"]), tr["y"])
        prob_val = clf.predict_proba(feats_all.transform(va["X"], va["age"], p["k"]))[:, 1]
        joblib.dump(dict(features=feats_all, classifier=clf, params=p, cutoff=cutoff,
                         experiment=exp), out / "models" / f"{m}.joblib")
        oof_out[f"prob_{m}"] = prob_oof
        val_out[f"prob_{m}"] = prob_val

        val = dict(roc_auc=roc_auc_score(va["y"], prob_val),
                   roc_auc_ci=bootstrap_ci(va["y"], prob_val, roc_auc_score, n_boot, seed),
                   pr_auc=average_precision_score(va["y"], prob_val),
                   pr_auc_ci=bootstrap_ci(va["y"], prob_val, average_precision_score,
                                          n_boot, seed),
                   **threshold_metrics(va["y"], prob_val, cutoff))
        metrics["models"][m] = dict(
            params={k: (v if isinstance(v, str) else float(v)) for k, v in p.items()},
            cv_auc_mean=float(best["cv_auc_mean"]), cv_auc_sd=float(best["cv_auc_sd"]),
            oof_auc=float(roc_auc_score(tr["y"], prob_oof)), cutoff=cutoff,
            oof_at_cutoff=threshold_metrics(tr["y"], prob_oof, cutoff), val=val)

        v = val
        print(f"\n{m}: chosen {p}")
        print(f"  CV AUC {best['cv_auc_mean']:.3f} (SD {best['cv_auc_sd']:.3f}); "
              f"out-of-fold AUC {metrics['models'][m]['oof_auc']:.3f}; cut-off {cutoff:.3f}")
        print(f"  VALIDATION: ROC-AUC {v['roc_auc']:.3f} [{v['roc_auc_ci'][0]:.3f}, "
              f"{v['roc_auc_ci'][1]:.3f}], PR-AUC {v['pr_auc']:.3f} [{v['pr_auc_ci'][0]:.3f}, "
              f"{v['pr_auc_ci'][1]:.3f}] (chance {va['y'].mean():.3f})")
        print(f"  sens {v['sensitivity']:.3f}, spec {v['specificity']:.3f}, "
              f"precision {v['precision']:.3f}, F1 {v['f1']:.3f}, confusion {v['confusion']}")

    oof_out.to_csv(out / "oof_train.csv", index=False)
    val_out.to_csv(out / "val_predictions.csv", index=False)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(f"\nSaved results to {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
