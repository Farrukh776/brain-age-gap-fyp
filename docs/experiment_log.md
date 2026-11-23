# Experiment Log

One row per run. Never delete rows; failed and negative runs stay.
Use validation metrics until a pipeline is frozen. Mark test-set runs clearly.

## Brain age
| Date | Who | Run ID | Model | Config | Git commit | Split | MAE | RMSE | r | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-23 | Farrukh + Claude | ba-ref | Mean training age (reference) | - | - | val (140) | 6.15 | - | - | Always predicts the mean training age; floor any model must beat |
| 2026-11-23 | Farrukh + Claude | ba-base-1 | Ridge, 4 mm voxels, PCA 800, alpha 1e4 | brain_age.yaml (grid max PCA 800, SVR C from 1) | 80a0ff4 (grid before extension) | val (140) | 3.80 | 4.90 | 0.708 | Grouped 5-fold CV on 1,755 train scans: MAE 4.05. Best at the largest PCA size, so the grid was extended |
| 2026-11-23 | Farrukh + Claude | ba-base-1 | SVR RBF, PCA 800, C 1, eps 2 | as above | as above | val (140) | 4.25 | 5.39 | 0.636 | CV MAE 4.70; worse than ridge at every PCA size |
| 2026-11-23 | Farrukh + Claude | ba-base-2 | **Ridge, 4 mm voxels, PCA 1200, alpha 1e4** | brain_age.yaml | 80a0ff4 | val (140) | **3.79** | 4.88 | 0.711 | CV MAE 4.04 (plateau vs run 1). Mean gap +0.09; gap-vs-age slope -0.47 (old under-, young over-predicted; bias correction is the next roadmap item). Chosen baseline (decision 24) |
| 2026-11-23 | Farrukh + Claude | ba-base-2 | SVR RBF, PCA 800, C 1, eps 2 | brain_age.yaml | 80a0ff4 | val (140) | 4.25 | 5.39 | 0.636 | CV MAE 4.70; C 0.3 did not help |

## Progression (E1-E5)
| Date | Who | Run ID | Exp | Model | Config | Git commit | Split | ROC-AUC | PR-AUC | Sens | Spec | F1 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
