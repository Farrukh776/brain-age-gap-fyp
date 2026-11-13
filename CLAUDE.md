# Brain Age Gap FYP

Group final year project (3 members, FAST-NUCES Karachi). Research question: does MRI-derived
Brain Age Gap add predictive value beyond chronological age for MCI-to-AD progression (ADNI, T1 MRI)?

## Start of every session

1. Read the top entry of `docs/SESSION_LOG.md` (where the last session stopped, what is next).
2. Read `docs/ROADMAP.md` (which step we are on) and `docs/PROJECT_CONTEXT.md` (scope and rules).
3. Check `docs/decisions.md` before proposing anything it already settles.

## End of every session

1. Add a new entry at the top of `docs/SESSION_LOG.md` (done, next, open questions).
2. Tick finished items in `docs/ROADMAP.md`; add new choices to `docs/decisions.md`.
3. Log model runs in `docs/experiment_log.md`. Then commit and push.

## How to work with the user

- Guide step by step, including basic steps. Explain what each command does before it is run.
- One roadmap step at a time, in order. Do not skip ahead.

## Scope

- **FYP-I (now):** cohort, preprocessing, brain-age model, BAG computation, E1 (MRI only),
  E2 (MRI + age). Ends when E2 is complete.
- **FYP-II (later):** E3-E5, statistics, XAI, calibration, integration. Do not build these yet.

## Hard rules

- Never commit ADNI data, subject IDs, tables, scans or model files. Never weaken `.gitignore`.
- Never print or paste subject-level rows into docs, commit messages or logs.
- Subject-level splits only; no subject in more than one of train / val / test.
- Do not touch the test set until a pipeline is frozen. Tune on train + validation.
- Brain-age bias correction is fitted on the training set only.
- E1 and E2 differ only by the age column: same subjects, splits, seeds, models.
- Cohort rules in `docs/cohort_definition.md` are fixed before any model is trained.
- Research prototype only; make no clinical claims.

## Conventions

- Python 3.11, conda env `fyp`, install with `pip install -r requirements.txt`.
- Code goes in `src/<stage>/`; settings in `configs/*.yaml` (no hard-coded paths); seed 42.
- Data lives in `data/` and results in `outputs/`, both local only.
- Git: branch `feature/<name>`, small commits, no direct commits to `main` once teammates join.
