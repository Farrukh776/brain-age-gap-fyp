# Project Context

Single source of truth for what this project is. Read this first. Summarised from the approved
FYP proposal (1 September 2026). Update it when a decision in `decisions.md` changes anything here.

## 1. One-paragraph summary

We test whether MRI-derived Brain Age Gap (BAG) adds predictive value beyond chronological age
for predicting which people with mild cognitive impairment (MCI) later progress to Alzheimer's
disease (AD). A brain-age model is trained on cognitively unimpaired (CN) participants, BAG is
computed for MCI participants, and nested progression models (E1-E5) isolate BAG's contribution.
Dataset: ADNI, T1-weighted structural MRI. Output: a reproducible research prototype, not a
clinical system.

## 2. Research questions

- **Primary:** Does BAG add predictive information beyond chronological age for MCI-to-AD
  progression? (Answered by E4 vs E2.)
- **Secondary:** Do selected clinical variables help (E5 vs E4)? Are explanations anatomically
  plausible and stable? Are predicted probabilities calibrated?

A negative result (BAG adds nothing) is a valid finding and will be reported.

## 3. Pipeline

```
ADNI T1 MRI -> Preprocessing (QC, registration, normalisation)
            -> Brain-age model (baseline ML + compact 3D CNN), trained on CN
            -> BAG = predicted age - chronological age, for MCI subjects
            -> MCI progression prediction (E1-E5)
            -> Trust layer: XAI + calibration
```

## 4. Experiments

| ID | Inputs | Purpose | Comparison | Phase |
|---|---|---|---|---|
| E1 | MRI-derived features | Imaging-only baseline | Reference | FYP-I |
| E2 | MRI + chronological age | Effect of adding age | E2 vs E1 | FYP-I |
| E3 | MRI + BAG | Effect of adding BAG | E3 vs E1 | FYP-II |
| E4 | MRI + age + BAG | BAG beyond age (primary) | E4 vs E2 | FYP-II |
| E5 | MRI + age + BAG + clinical | Limited multimodal (optional) | E5 vs E4 | FYP-II |

## 5. FYP-I boundary

FYP-I ends with E2 complete. Work is done in this order:

1. ADNI access
2. Repository and environment
3. Cohort definition and subject-level splits
4. MRI preprocessing
5. Brain-age model (baseline, then 3D CNN, then age-bias correction)
6. BAG computation for MCI subjects
7. E1
8. E2 and the E2 vs E1 comparison
9. FYP-I report and presentation

BAG is produced and described in FYP-I but is not used as a model input until E3 (FYP-II).

## 6. Non-negotiable rules

1. **Subject-level splits.** No participant appears in more than one of train / validation / test.
2. **Test set untouched** until a pipeline is frozen. Tune on train + validation only.
3. **Cohort rules written before modelling** (`cohort_definition.md`).
4. **Bias correction uses training-derived parameters only.**
5. **E1 vs E2 differ in exactly one thing:** the age column. Same subjects, splits, seeds, model family.
6. **No ADNI data or subject IDs in git** or in any published output.
7. **Accuracy is never the only metric** (classes may be imbalanced).
8. **No clinical claims.**

## 7. Evaluation

- **Brain age:** MAE, RMSE, Pearson/Spearman correlation, age-bias analysis.
- **Progression:** ROC-AUC, PR-AUC, precision, recall (sensitivity), specificity, F1, confusion
  matrix, confidence intervals where feasible.
- **Calibration (FYP-II):** calibration curves, Brier score, expected calibration error.
- **XAI (FYP-II):** SHAP / Grad-CAM / permutation importance, judged on anatomical plausibility
  and stability.

## 8. In scope / out of scope

**In:** T1 MRI; ADNI; brain-age regression; BAG; MCI-to-AD prediction from baseline data;
ablations; XAI; calibration; optional limited clinical fusion.

**Out:** clinical diagnosis or deployment; new data collection; a novel architecture as the main
contribution; all modalities at once; large-scale genetics; longitudinal BAG; dependence on
external datasets.

## 9. Team streams

| Stream | Responsibilities |
|---|---|
| Brain Age AI | Preprocessing, cohort QC support, brain-age baseline, 3D CNN, bias correction, BAG |
| Progression AI | MCI cohort, conversion labels, features, progression models, ablations, statistics |
| Trustworthy AI & Integration | XAI, calibration, experiment tracking, final integration |

All members share literature review, evaluation, report writing and presentations.

## 10. Tools

Python, PyTorch, scikit-learn, NiBabel, Nilearn, SimpleITK, SHAP, Captum, NumPy, pandas, SciPy,
Matplotlib, Plotly, Git/GitHub, YAML configs.

## 11. Key references

1. Kang et al., 2026, Computerized Medical Imaging and Graphics. Interpretable MRI framework; MCI conversion AUC 0.750 on ADNI.
2. Liu et al., 2023, IEEE TNSRE. Brain-age based conversion risk; age gap associated with MCI conversion.
3. Lu and Li, 2024, J. Central Nervous System Disease. Brain-age models for MCI converters; AUC about 0.70.
4. Jin et al., 2025, Alzheimer's & Dementia: DADM. Longitudinal BAG in MCI (out of our scope).
5. Li et al., 2026, J. NeuroEngineering and Rehabilitation. SHAP and Grad-CAM in EEG-based AD.
6. Viswan et al., 2026, Brain Informatics. Review of multimodal fusion and explainability in AD.
7. Kim et al., 2025, npj Aging. Deep learning brain age for routine clinical MRI.
