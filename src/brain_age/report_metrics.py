"""Report the frozen brain-age model's metrics from saved predictions (no model is run).

Run from the repo root:  python -m src.brain_age.report_metrics

Reads the predictions that predict.py already saved (decision 31): CN validation and the one CN
test run. Computing extra metrics from those saved predictions is not a new use of the test set:
nothing is predicted again and nothing can be changed by the result.
Reports MAE, RMSE, Pearson r and Spearman rho (PROJECT_CONTEXT section 7) for the final model
(5-fold CNN average, uncorrected brain age) and, on test, cnn-1 and ridge for comparison.
Prints summary numbers only.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr


def metrics(age: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - age
    return dict(n=len(age), mae=np.abs(err).mean(), rmse=np.sqrt((err ** 2).mean()),
                pearson_r=pearsonr(age, pred)[0], spearman_rho=spearmanr(age, pred)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/brain_age.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    final = Path(cfg["paths"]["out_dir"]) / "final"

    rows = []
    for split in ("val", "test"):
        t = pd.read_csv(final / f"cn_{split}_predictions.csv")
        cols = {"final (5-fold CNN average)": "brain_age"}
        if split == "test":
            cols.update({"cnn-1 (comparison)": "pred_cnn1", "ridge (comparison)": "pred_ridge"})
        for name, col in cols.items():
            rows.append(dict(split=f"CN {split}", model=name, **metrics(t["age"].to_numpy(),
                                                                        t[col].to_numpy())))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
