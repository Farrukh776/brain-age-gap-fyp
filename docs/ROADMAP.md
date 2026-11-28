# FYP-I Roadmap

Tick items as they are completed. Steps are done in order.

## Step 1: ADNI access
- [x] ADNI access granted
- [ ] All three members can log in to the IDA (or access arranged via supervisor)

## Step 2: Repository
- [x] GitHub repo created (public, see decision 3), starter files pushed
- [ ] Teammates and supervisor added as collaborators
- [ ] Every member has cloned and pushed one test branch

## Step 3: Environment
- [ ] Conda environment `fyp` works on all machines (Farrukh: done 2026-11-13; Anfas, Asher: pending)
- [x] Storage location for scans decided (see decision 6)
- [ ] GPU access confirmed (university lab GPU available; model, storage and booking rules to confirm)

## Step 4: Cohort and splits
- [x] Clinical tables downloaded (diagnosis, demographics, MMSE, registry, data dictionary;
      ADNIMERGE skipped, it is now an R package only)
- [ ] `docs/cohort_definition.md` filled in and agreed with supervisor
      (filled in 2026-11-14, decisions 7 to 12; supervisor review pending)
- [x] Brain-age cohort (CN) table built
- [x] Progression cohort (MCI) table built with progressor / stable labels
- [x] Subject-level 70/15/15 splits saved for both cohorts
- [x] Verified no subject overlaps between cohorts or splits
- [x] Class balance and exclusion counts reported (in `docs/cohort_definition.md`)

## Step 5: MRI preprocessing
- [x] Baseline T1 scans downloaded for both cohorts (1,401 of 1,401, checked 2026-11-16)
- [x] Preprocessing tool chosen and recorded in `decisions.md` (decision 13)
- [x] Pilot on 10 scans, visually checked
- [x] Full batch run (1,401 baseline scans, every scan checked by eye, 2026-11-16)
- [x] Failures and parameters logged in `docs/preprocessing_log.md` (1,393 of 1,401 subjects usable)
- [x] Later CN training visits (1,110 scans) downloaded, preprocessed and checked by eye
      (2026-11-18: 1,101 usable; brain-age training set 1,758 scans)
- [x] Final check of every scan before modelling (2026-11-22: 21 tilted scans fixed or removed,
      4 duplicate enrolments removed; 1,388 subjects, 2,489 scans; decisions 20 and 21)

## Step 6 preparation
- [x] Feature source for the SVR / ridge baseline decided: own image features (decision 19)
- [x] Registration template figure for the report (`docs/figures/mni_template.png`)

## Step 6: Brain-age model
- [x] Features extracted for CN subjects (2026-11-23: 4 mm voxels, 29,520 per scan, all 2,489
      scans incl. MCI; decision 22)
- [x] Baseline (SVR / ridge) trained; MAE, RMSE, correlation on validation
      (2026-11-23: ridge, val MAE 3.79, RMSE 4.88, r 0.71; decision 24)
- [x] 3D CNN data loader tested on 20 scans (2026-11-23: `src/brain_age/data.py`, 2 mm input
      80 x 96 x 80, flip + shift augmentation on train only; all checks passed)
- [x] 3D CNN trained; MAE, RMSE, correlation on validation
      (2026-11-23: run cnn-1 on Colab, val MAE 3.45, RMSE 4.33, r 0.79; beats ridge 3.79)
- [x] Age-bias plot; correction fitted on training set only
      (2026-11-27: linear, on out-of-fold training predictions; 5 fold CNNs; validation gap slope
      -0.37 -> +0.01; decision 30)
- [x] Final model chosen on validation, reported once on test
      (2026-11-27: 5-fold CNN average, decision 31; test MAE 3.57, RMSE 4.57, r 0.74)

## Step 7: BAG computation
- [x] Brain age predicted for every MCI subject
      (2026-11-27: 453 subjects, frozen model of decision 31, `predict.py --set mci`)
- [x] Bias correction applied
      (training-fitted; MCI corrected BAG mean +2.52, CN test +0.15; slope vs age -0.02)
- [x] BAG table saved (ID, age, predicted age, BAG, label)
      (`src/brain_age/bag_table.py`; local `outputs/brain_age/bag/bag_table.csv`)
- [x] BAG distribution: progressors vs stable, with statistical test
      (2026-11-27: MCI train + val, 384; corrected BAG +4.15 vs +1.75, difference +2.40 y
      [1.65, 3.17], Hedges' g 0.68, Welch p 1e-9; decision 32. Repeat on all 453 after E1-E5)

## Step 8: E1 (MRI only)
- [x] MRI feature table for MCI cohort
      (2026-11-28: 4 mm voxels of decision 22, 453 subjects x 29,520; decision 33;
      `src/features/mci_features.py`, local `data/features/mci_voxels_4mm.npy`)
- [ ] Logistic regression + one tree model, 5-fold CV on train
- [ ] Validation metrics logged in `docs/experiment_log.md`

## Step 9: E2 (MRI + age)
- [ ] Same pipeline, splits and seeds as E1, plus age
- [ ] Validation metrics logged
- [ ] Both pipelines frozen, then evaluated on test
- [ ] E2 vs E1 comparison (DeLong or bootstrap) with confidence intervals

## Step 10: FYP-I deliverables
- [ ] Report: literature review, cohort flow diagram, preprocessing, brain age, BAG, E1/E2
- [ ] FYP-II plan (E3-E5, XAI, calibration)
- [ ] Presentation slides
