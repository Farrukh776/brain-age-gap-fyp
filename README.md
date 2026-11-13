# Brain Age Gap and Trustworthy Multimodal AI for Predicting Alzheimer's Disease Progression

Final Year Project, BS (AI), FAST School of Computing, FAST-NUCES Karachi. Session Fall 2026.

**Supervisor:** Dr. Nasir Uddin

| Member | Reg. # | Stream |
|---|---|---|
| Syed Anfas | 23K-0574 | TBD |
| Asher Ahmed | 23K-0511 | TBD |
| Farrukh | 23K-0776 | TBD |

Streams (from the proposal): Brain Age AI, Progression AI, Trustworthy AI & Integration.

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
tests/
```

## Team workflow

1. Never commit directly to `main`.
2. Create a branch: `git checkout -b feature/<short-name>`.
3. Commit small, with clear messages.
4. Push and open a Pull Request; one teammate reviews before merging.
5. Record every experiment run in [docs/experiment_log.md](docs/experiment_log.md).
6. Record every design decision in [docs/decisions.md](docs/decisions.md).
