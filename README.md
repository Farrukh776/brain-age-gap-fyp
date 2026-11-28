# Brain Age Gap and Trustworthy Multimodal AI for Predicting Alzheimer's Disease Progression

Final Year Project, BS (AI), FAST School of Computing, FAST-NUCES Karachi. Session Fall 2026.

**Supervisor:** Dr. Nasir Uddin

| Member | Reg. # | Stream |
|---|---|---|
| Syed Anfas | 23K-0574 | All (shared) |
| Asher Ahmed | 23K-0511 | All (shared) |
| Farrukh | 23K-0776 | All (shared) |

Streams (from the proposal): Brain Age AI, Progression AI, Trustworthy AI & Integration.
The team works on all three streams together rather than assigning one per member.

## Research question

Does MRI-derived Brain Age Gap (BAG) add predictive information beyond chronological age
for MCI-to-AD progression, under controlled patient-level experiments?

`BAG = predicted brain age - chronological age`

This is a research prototype. It is not a clinical diagnostic tool.

## Scope by phase

| Phase | Covers |
|---|---|
| **FYP-I** | Cohort, MRI preprocessing, brain-age model, BAG computation, E1 (MRI only), E2 (MRI + age) |
| **FYP-II** | E3 (MRI + BAG), E4 (MRI + age + BAG), E5 (optional clinical fusion), statistics, XAI, calibration, integration, thesis |

Full context: [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md). Current status: [docs/ROADMAP.md](docs/ROADMAP.md).

## Setup

```
git clone <repo-url>
cd <repo-folder>
conda create -n fyp python=3.11
conda activate fyp
pip install -r requirements.txt
```

## Data

Data comes from ADNI and is **not** in this repository. Each member downloads it with their own
ADNI account and places it under `data/` (see [data/README.md](data/README.md)).
Never commit scans, clinical tables, or anything containing subject IDs.

## Reproducing FYP-I

Run every command from the repo root with the `fyp` environment active. Settings live in
`configs/*.yaml`; the decision numbers refer to [docs/decisions.md](docs/decisions.md). Steps that
use a test set are run **once**, after the pipeline is frozen; the scripts refuse a second run.

1. **Cohorts and splits** (step 4; `docs/cohort_definition.md`)
   ```
   python -m src.cohort.inspect_tables
   python -m src.cohort.build_cohorts
   python -m src.cohort.make_splits
   ```
2. **MRI preprocessing and QC** (step 5; `docs/preprocessing_log.md`, decisions 13-21)
   ```
   python -m src.preprocessing.make_download_lists      # then download the scans from ADNI
   python -m src.preprocessing.preprocess --scans <scan list csv>
   python -m src.preprocessing.qc_sheets                # every scan checked by eye
   python -m src.preprocessing.record_review --sheets <n ...>
   python -m src.preprocessing.check_tilt --workers 6
   python -m src.preprocessing.make_training_scans
   python -m src.preprocessing.verify_processed --workers 6
   ```
3. **Brain-age model** (step 6; decisions 22-31; CNN training on a GPU: `docs/gpu_training_guide.md`)
   ```
   python -m src.brain_age.extract_features --workers 4
   python -m src.brain_age.train_baseline                      # ridge / SVR baseline
   python -m src.brain_age.cache build --cache-dir <folder>    # 2 mm arrays for the GPU machine
   python -m src.brain_age.train_cnn --run cnn-1               # and --fold 0..4 (run cnn-cv)
   python -m src.brain_age.bias_plot
   python -m src.brain_age.bias_correction --cv-run cnn-cv
   python -m src.brain_age.predict --set cn_val                # must reproduce validation
   python -m src.brain_age.predict --set cn_test --confirm-test   # done once (decision 31)
   python -m src.brain_age.report_metrics                      # MAE, RMSE, Pearson, Spearman
   ```
4. **BAG** (step 7; decision 32)
   ```
   python -m src.brain_age.predict --set mci
   python -m src.brain_age.bag_table
   python -m src.brain_age.bag_stats          # MCI train + validation only, until E1-E5 are frozen
   ```
5. **E1 and E2** (steps 8-9; decisions 33-36)
   ```
   python -m src.features.mci_features
   python -m src.progression.train --config configs/e1.yaml
   python -m src.progression.train --config configs/e2.yaml
   python -m src.progression.evaluate --set val                   # checks hashes, dry run
   python -m src.progression.evaluate --set test --confirm-test   # done once (decision 36)
   ```
6. **Tests**: `python -m pytest` (about 20 s). Tests that need ADNI data skip themselves when the
   data is not on the machine.

Results of every run: [docs/experiment_log.md](docs/experiment_log.md).

## Repository layout

```
configs/        One YAML file per stage or experiment
data/           Local only, ignored by git
docs/           Context, roadmap, cohort definition, logs, meeting notes
notebooks/      Exploration only; final code lives in src/
outputs/        Local only, ignored by git (models, figures, metrics)
src/
  cohort/         Build cohorts and subject-level splits
  preprocessing/  MRI preprocessing and QC
  features/       MRI feature extraction
  brain_age/      Brain-age baseline, 3D CNN, bias correction, BAG
  progression/    E1-E5 training and evaluation
  utils/          Shared helpers (config, seeding, metrics)
tests/          pytest checks (configs, metrics, subject-level data rules)
```

## Team workflow

1. Never commit directly to `main`.
2. Create a branch: `git checkout -b feature/<short-name>`.
3. Commit small, with clear messages.
4. Push and open a Pull Request; one teammate reviews before merging.
5. Record every experiment run in [docs/experiment_log.md](docs/experiment_log.md).
6. Record every design decision in [docs/decisions.md](docs/decisions.md).
