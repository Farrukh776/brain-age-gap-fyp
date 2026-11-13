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
- [ ] GPU access confirmed (needed for the 3D CNN; Farrukh's laptop has no NVIDIA GPU)

## Step 4: Cohort and splits
- [ ] Clinical tables downloaded (diagnosis, demographics, MMSE, merged summary)
- [ ] `docs/cohort_definition.md` filled in and agreed with supervisor
- [ ] Brain-age cohort (CN) table built
- [ ] Progression cohort (MCI) table built with progressor / stable labels
- [ ] Subject-level 70/15/15 splits saved for both cohorts
- [ ] Verified no subject overlaps between cohorts or splits
- [ ] Class balance and exclusion counts reported

## Step 5: MRI preprocessing
- [ ] Baseline T1 scans downloaded for both cohorts
- [ ] Preprocessing tool chosen and recorded in `decisions.md`
- [ ] Pilot on 10 scans, visually checked
- [ ] Full batch run
- [ ] Failures and parameters logged in `docs/preprocessing_log.md`

## Step 6: Brain-age model
- [ ] Features extracted for CN subjects
- [ ] Baseline (SVR / ridge) trained; MAE, RMSE, correlation on validation
- [ ] 3D CNN data loader tested on 20 scans
- [ ] 3D CNN trained; MAE, RMSE, correlation on validation
- [ ] Age-bias plot; correction fitted on training set only
- [ ] Final model chosen on validation, reported once on test

## Step 7: BAG computation
- [ ] Brain age predicted for every MCI subject
- [ ] Bias correction applied
- [ ] BAG table saved (ID, age, predicted age, BAG, label)
- [ ] BAG distribution: progressors vs stable, with statistical test

## Step 8: E1 (MRI only)
- [ ] MRI feature table for MCI cohort
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
