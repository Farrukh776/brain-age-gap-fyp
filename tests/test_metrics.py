"""DeLong test, cut-off and threshold metrics on simulated data (no ADNI data needed)."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.progression.evaluate import delong, paired_bootstrap_diff
from src.progression.train import threshold_metrics, youden_cutoff


def simulated(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype(int)
    a = y * 0.8 + rng.normal(size=n)
    b = np.round(0.7 * a + 0.5 * rng.normal(size=n) + 0.3 * y, 1)   # rounded, so it has ties
    return y, a, b


def test_delong_auc_matches_sklearn_with_ties():
    y, a, b = simulated()
    aucs, _ = delong(y, np.vstack([a, b]))
    assert np.allclose(aucs, [roc_auc_score(y, a), roc_auc_score(y, b)])


def test_delong_variance_matches_bootstrap():
    y, a, b = simulated()
    _, cov = delong(y, np.vstack([a, b]))
    rng = np.random.default_rng(1)
    single, diffs = [], []
    while len(single) < 2000:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            single.append(roc_auc_score(y[i], a[i]))
            diffs.append(single[-1] - roc_auc_score(y[i], b[i]))
    assert np.sqrt(cov[0, 0]) == pytest.approx(np.std(single), rel=0.1)
    assert np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]) == pytest.approx(np.std(diffs), rel=0.1)


def test_identical_models_have_zero_difference():
    y, a, _ = simulated()
    aucs, cov = delong(y, np.vstack([a, a]))
    assert aucs[0] == aucs[1]
    assert cov[0, 0] + cov[1, 1] - 2 * cov[0, 1] == pytest.approx(0, abs=1e-12)
    lo, hi = paired_bootstrap_diff(y, a, a, 200, 42)
    assert lo == hi == 0


def test_threshold_metrics_by_hand():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.7, 0.3, 0.1, 0.05])
    m = threshold_metrics(y, p, 0.5)          # predicted positive: 0.9, 0.6, 0.7
    assert m["confusion"] == dict(tn=3, fp=1, fn=1, tp=2)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["precision"] == pytest.approx(2 / 3)


def test_youden_cutoff_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    c = youden_cutoff(y, p)
    assert 0.3 < c <= 0.8
    assert threshold_metrics(y, p, c)["confusion"] == dict(tn=3, fp=0, fn=0, tp=2)
