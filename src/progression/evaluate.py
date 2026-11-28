"""Evaluate the frozen E1 / E2 pipelines and compare them (step 9, decision 36).

Dry run on validation (allowed any time; checks the code):
    python -m src.progression.evaluate --set val
The one MCI test run (only after the pipelines are frozen in configs/progression_test.yaml):
    python -m src.progression.evaluate --set test --confirm-test

Safety checks, in order:
  1. Every model file's SHA-256 must match configs/progression_test.yaml (test: required;
     validation dry run: hashes are printed if not frozen yet).
  2. The saved models must reproduce their saved validation predictions (within 1e-6).
  3. The test set is read only with --confirm-test and only if no test result exists yet.
Metrics per experiment and model: ROC-AUC (DeLong 95% CI), PR-AUC (bootstrap CI), and
sensitivity, specificity, precision, F1 and confusion matrix at the cut-off fixed on training.
Comparison E2 vs E1 per model: AUC difference, paired DeLong test (two-sided) with its 95% CI,
and a paired bootstrap 95% CI. Writes <out_dir>/<set>_metrics.json and <set>_predictions.csv
(local only). Prints summary numbers only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from src.progression.train import bootstrap_ci, git_commit, threshold_metrics


# ---------------------------------------------------------------- DeLong

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y: np.ndarray, preds: np.ndarray):
    """AUCs and their covariance for k models scored on the same subjects (DeLong 1988,
    fast algorithm of Sun and Xu 2014). preds: shape (k, n). Returns (aucs, cov)."""
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01).reshape(k, k)
    sy = np.cov(v10).reshape(k, k)
    return aucs, sx / m + sy / n


def delong_ci(auc: float, var: float, alpha: float):
    z = norm.ppf(1 - alpha / 2)
    return [float(max(0.0, auc - z * np.sqrt(var))), float(min(1.0, auc + z * np.sqrt(var)))]


def paired_bootstrap_diff(y, p_a, p_b, n: int, seed: int):
    """95% CI of AUC(a) - AUC(b), resampling subjects (same resample for both models)."""
    rng = np.random.default_rng(seed)
    diffs = []
    while len(diffs) < n:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


# ---------------------------------------------------------------- data

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_split(exp_cfg: dict, split: str):
    paths = exp_cfg["paths"]
    index = pd.read_csv(paths["features_index"])
    X = np.load(paths["features"], mmap_mode="r")
    cohort = pd.read_csv(paths["cohort_table"])[["RID", "age", "label"]]
    splits = pd.read_csv(paths["splits"]).rename(columns={"split": "split_table"})
    t = (index.reset_index().merge(cohort, on="RID", how="left", validate="one_to_one")
         .merge(splits, on="RID", how="left", validate="one_to_one"))
    if (t["split"] != t["split_table"]).any() or t[["age", "label"]].isna().any().any():
        raise SystemExit("feature index, cohort and splits disagree")
    s = t[t["split"] == split].reset_index(drop=True)
    return dict(rid=s["RID"].to_numpy(),
                X=np.asarray(X[s["index"].to_numpy()], dtype=np.float32),
                age=s["age"].to_numpy(dtype=float).reshape(-1, 1),
                y=s["label"].to_numpy(dtype=int))


def predict(model: dict, d: dict) -> np.ndarray:
    k = int(model["params"]["k"])
    return model["classifier"].predict_proba(model["features"].transform(d["X"], d["age"], k))[:, 1]


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/progression_test.yaml")
    parser.add_argument("--set", choices=["val", "test"], default="val")
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed, alpha, n_boot = cfg["seed"], cfg["alpha"], cfg["bootstrap_iterations"]
    models = [cfg["primary_model"]] + cfg["secondary_models"]
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    if args.set == "test":
        if not args.confirm_test:
            raise SystemExit("the MCI test set is used once; add --confirm-test to run it")
        if (out / "test_metrics.json").exists():
            raise SystemExit("a test result already exists; the test set must not be used again")

    # 1. hashes
    loaded, hashes, frozen = {}, {}, True
    for exp, exp_path in cfg["experiments"].items():
        exp_cfg = yaml.safe_load(open(exp_path))
        if exp_cfg["experiment"] != exp:
            raise SystemExit(f"{exp_path} is not {exp}")
        for m in models:
            path = Path(exp_cfg["paths"]["out_dir"]) / "models" / f"{m}.joblib"
            digest = sha256(path)
            hashes[f"{exp}/{m}"] = digest
            expected = (cfg.get("frozen") or {}).get(exp, {}).get(m)
            if expected is None:
                frozen = False
            elif expected != digest:
                raise SystemExit(f"{exp}/{m}: model file differs from the frozen hash")
            loaded[(exp, m)] = (joblib.load(path), exp_cfg)
    if not frozen:
        if args.set == "test":
            raise SystemExit("pipelines are not frozen (hashes missing); the test set stays closed")
        print("Not frozen yet. SHA-256 of the model files:")
        for key, digest in hashes.items():
            print(f"  {key}: {digest}")
    else:
        print("All model files match the frozen hashes.")

    # 2. saved models reproduce their saved validation predictions (up to floating-point
    #    rounding in the matrix products, about 1e-8 on probabilities)
    worst = 0.0
    for (exp, m), (model, exp_cfg) in loaded.items():
        val = load_split(exp_cfg, "val")
        saved = pd.read_csv(Path(exp_cfg["paths"]["out_dir"]) / "val_predictions.csv")
        if not np.array_equal(saved["RID"].to_numpy(), val["rid"]):
            raise SystemExit(f"{exp}: validation subjects differ from the saved predictions")
        diff = np.abs(predict(model, val) - saved[f"prob_{m}"].to_numpy()).max()
        if diff > 1e-6:
            raise SystemExit(f"{exp}/{m}: does not reproduce its validation predictions ({diff:.1e})")
        worst = max(worst, diff)
    print(f"Saved models reproduce their validation predictions (largest difference {worst:.1e}).")

    # 3. evaluate the requested set
    data = {exp: load_split(exp_cfg, args.set)
            for exp, exp_cfg in {e: c for (e, _), (_, c) in loaded.items()}.items()}
    exps = list(cfg["experiments"])
    ref = data[exps[0]]
    for e in exps[1:]:
        if not (np.array_equal(data[e]["rid"], ref["rid"]) and np.array_equal(data[e]["y"], ref["y"])):
            raise SystemExit("experiments see different subjects or labels")
    y = ref["y"]
    print(f"\nSet: {args.set} ({len(y)} subjects, {y.sum()} progressors)"
          + ("  ** MCI TEST SET, USED ONCE **" if args.set == "test" else ""))

    result = dict(set=args.set, n=int(len(y)), n_progressors=int(y.sum()),
                  git_commit=git_commit(), model_sha256=hashes, frozen=frozen,
                  experiments={}, comparisons={})
    pred_out = pd.DataFrame({"RID": ref["rid"], "label": y})
    probs = {}
    for (exp, m), (model, _) in loaded.items():
        p = predict(model, data[exp])
        probs[(exp, m)] = p
        pred_out[f"{exp}_{m}"] = p
        auc, cov = delong(y, p[None, :])
        r = dict(roc_auc=float(auc[0]), roc_auc_ci_delong=delong_ci(auc[0], cov[0, 0], alpha),
                 pr_auc=float(average_precision_score(y, p)),
                 pr_auc_ci=bootstrap_ci(y, p, average_precision_score, n_boot, seed),
                 cutoff=float(model["cutoff"]), **threshold_metrics(y, p, model["cutoff"]))
        result["experiments"].setdefault(exp, {})[m] = r
        print(f"{exp} {m}: ROC-AUC {r['roc_auc']:.3f} [{r['roc_auc_ci_delong'][0]:.3f}, "
              f"{r['roc_auc_ci_delong'][1]:.3f}], PR-AUC {r['pr_auc']:.3f} "
              f"[{r['pr_auc_ci'][0]:.3f}, {r['pr_auc_ci'][1]:.3f}], sens {r['sensitivity']:.3f}, "
              f"spec {r['specificity']:.3f}, prec {r['precision']:.3f}, F1 {r['f1']:.3f}, "
              f"{r['confusion']}")

    a, b = cfg["comparison"]
    for m in models:
        aucs, cov = delong(y, np.vstack([probs[(a, m)], probs[(b, m)]]))
        diff = float(aucs[0] - aucs[1])
        var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
        z = diff / np.sqrt(var) if var > 0 else 0.0
        pval = float(2 * norm.sf(abs(z)))
        zc = norm.ppf(1 - alpha / 2)
        c = dict(primary=(m == cfg["primary_model"]), auc_diff=diff,
                 delong_ci=[diff - zc * np.sqrt(var), diff + zc * np.sqrt(var)],
                 delong_z=float(z), delong_p=pval,
                 bootstrap_ci=paired_bootstrap_diff(y, probs[(a, m)], probs[(b, m)], n_boot, seed))
        result["comparisons"][f"{a}_vs_{b}/{m}"] = c
        tag = "PRIMARY" if c["primary"] else "secondary"
        print(f"\n{a} vs {b}, {m} ({tag}): AUC difference {diff:+.4f}; DeLong 95% CI "
              f"[{c['delong_ci'][0]:+.4f}, {c['delong_ci'][1]:+.4f}], z {z:.2f}, p {pval:.3f}; "
              f"paired bootstrap 95% CI [{c['bootstrap_ci'][0]:+.4f}, {c['bootstrap_ci'][1]:+.4f}]")

    with open(out / f"{args.set}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    pred_out.to_csv(out / f"{args.set}_predictions.csv", index=False)
    print(f"\nSaved {out / (args.set + '_metrics.json')}")


if __name__ == "__main__":
    main()
