# Cohort Definition

**Status:** DECIDED by the team on 2026-11-14 (decisions 7 to 12 in `decisions.md`), based on the
counts from `src/cohort/inspect_tables.py`. Not yet reviewed by the supervisor; show him this
file before any model is trained. Any later change needs a new entry in `decisions.md`.

## Data source
- Dataset: ADNI. Phases: ADNI1, GO, 2, 3 and 4 for the brain-age cohort; ADNI1, GO, 2 and 3 for
  the progression cohort (ADNI4 MCI subjects enrolled too recently for a 36-month outcome).
- Tables used (ADNI Study Files, downloaded 2026-11-14, file version 13Nov2026), kept in
  `data/raw/tables`:
  - `DXSUM_13Nov2026.csv` (diagnosis), `PTDEMOG_13Nov2026.csv` (demographics),
    `MMSE_13Nov2026.csv`, `REGISTRY_13Nov2026.csv` (visits), `DATADIC_13Nov2026.csv` (dictionary)
  - `MRIMETA_13Nov2026.csv`, `MRI3META_13Nov2026.csv` (scan information, field strength)
  - `Clinical_T1w_Imaging_Cohort_Manifest_13Nov2026.csv` (list of T1 images)
  - `UCSFFSX7_13Nov2026.csv` (FreeSurfer 7.x regional measures; use decided in step 6)
  - ADNIMERGE is not used (now distributed as an R package only).
- Imaging: T1-weighted structural MRI (MPRAGE / IR-SPGR family), original images as listed in
  the manifest, **3 T only**.
  Reason: 43% of MCI progressors but only 10% of CN subjects were scanned at 1.5 T. Mixing field
  strengths would let BAG reflect the scanner. 3 T only removes this without any correction step.
  Cost: fewer subjects (about 148 progressors instead of 261).

## A. Brain-age cohort (cognitively unimpaired)
Inclusion:
- Diagnosis CN at baseline (the subject's first visit with a recorded diagnosis)
- 3 T T1 scan matched to that visit (see Visit alignment) that passes QC
- Known chronological age at scan

Exclusion:
- CN subjects who are diagnosed MCI or AD at any later visit.
  Reason: the model should learn normal ageing; subjects with early undetected disease would
  make an "older-looking" brain seem normal and shrink BAG in the MCI cohort.
  The number excluded is reported. They are not moved into the MCI cohort.

Scans per subject: multiple visits in the training set (only visits where the subject is still
CN); one baseline scan per subject in validation and test. All scans of a subject stay in the
same split.

## B. Progression cohort (MCI)
Inclusion:
- Diagnosis MCI at baseline
- 3 T baseline T1 scan matched to that visit that passes QC
- Enrolled in ADNI1, GO, 2 or 3

Scans per subject: baseline only (required by the proposal; longitudinal BAG is out of scope).

Outcome definition:
- Follow-up window: 36 months from the baseline diagnosis
- **Progressor (label 1):** diagnosed with dementia at any visit within the window
- **Stable (label 0):** never diagnosed with dementia within the window, and has a diagnosis
  visit at 36 months or later

Exclusion:
- Diagnosed CN at any follow-up visit (reverters), unless already a progressor
- Follow-up shorter than the window without conversion
- Missing age or failed scan QC

## Visit alignment
- A scan is matched to the baseline diagnosis if it was acquired within 90 days of the baseline
  diagnosis date. If several qualify, the nearest is used; a repeat scan from the same session
  is only used if the first fails QC.

## Splits
- Subject-level. Ratio 70 / 15 / 15. Seed: 42. Created by `src/cohort/make_splits.py`.
- MCI cohort: stratified by label. CN cohort: stratified by baseline age band (five equal-sized
  bands), so every split covers the full age range.
- Split files are created once and never regenerated (the script refuses to overwrite them).
  Subjects that later fail scan QC are removed from their split; nobody is moved between splits.

## Implementation notes
- Built by `src/cohort/build_cohorts.py` from `configs/cohort.yaml`.
- Months since baseline use the scheduled visit code (`VISCODE2`: sc/bl = 0, m36 = 36), because
  visits rarely fall on the exact day; dates are the fallback.
- Diagnosis rows without an exam date take the date from the registry table.
- Field strength per scan session comes from the scan information tables (matched by date),
  then from the FreeSurfer table. 315 of 11,603 sessions stay unknown (subjects scanned at both
  strengths on the same day) and are not used; this affects at most 3 candidate subjects.
- One image per session: the first (non-repeat), non-accelerated acquisition when available.
  ADNI3 and ADNI4 only have accelerated scans, so those are used there.
- Non-T1 images in the T1 manifest (localizer, scout, survey, calibration, FLAIR, T2) are never
  chosen (decision 15, 2026-11-16). This replaced the image for 2 sessions (1 CN, 1 MCI) with
  the T1 from the same session; subjects, sessions, counts and splits did not change.
- Age is scan date minus date of birth (ADNI gives month and year; the 15th is assumed).

## Counts (built and split 2026-11-14, before scan QC)
| Cohort | Start | After exclusions | Train | Val | Test |
|---|---|---|---|---|---|
| CN (subjects) | 1593 | 947 | 662 | 142 | 143 |
| MCI stable | 1550 (all MCI) | 299 | 209 | 45 | 45 |
| MCI progressor | | 155 | 108 | 23 | 24 |

Progression cohort: 454 subjects, 34% progressors in every split.
Brain-age cohort: 2497 scans in total; the training subjects have 1776 scans across visits.
Validation and test use the baseline scan only (142 and 143 scans).

## Counts after scan QC (2026-11-16)
Every baseline scan was preprocessed and checked by eye (see `preprocessing_log.md`, decisions
16 and 17). 3 failed scans were replaced by another T1 from the same session; 8 subjects had
no usable scan and were removed from their split (nobody was moved).

| Cohort | Before QC | Failed scan QC | After QC | Train | Val | Test |
|---|---|---|---|---|---|---|
| CN (subjects) | 947 | 7 | 940 | 657 | 141 | 142 |
| MCI stable | 299 | 0 | 299 | 209 | 45 | 45 |
| MCI progressor | 155 | 1 | 154 | 107 | 23 | 24 |

Brain-age training scans after QC (2026-11-18): 657 baseline scans plus 1,101 of 1,110 later CN
visits = 1,758 scans (9 later visits failed scan QC and were dropped; no subject was removed).

## Final data after preprocessing and QC (2026-11-18)
Terms: a **subject** is one person; a **scan** (ADNI "image", one image ID) is one MRI. Each
visit contributes one scan. CN training subjects have one scan per CN visit; everyone else has
one baseline scan.

| Cohort | Split | Subjects | Scans | Notes |
|---|---|---|---|---|
| CN | Train | 657 | 1,758 | 657 baseline + 1,101 later visits; 1 to 11 scans per subject (324 have 1) |
| CN | Validation | 141 | 141 | baseline only |
| CN | Test | 142 | 142 | baseline only |
| MCI | Train | 316 | 316 | 209 stable, 107 progressors |
| MCI | Validation | 68 | 68 | 45 stable, 23 progressors |
| MCI | Test | 69 | 69 | 45 stable, 24 progressors |
| **Total** | | **1,393** | **2,494** | |

`data/processed` holds 2,517 files; the 23 extra are scans that failed QC (11 baseline, 3
same-session repeats, 9 later visits) and are never used. Which scan to use always comes from
`data/tables/cn_train_scans.csv` (CN training) and `data/tables/scan_qc_final.csv` (everything
else), never from the folder (decision 18).

## Exclusion log
| Cohort | Rule | Subjects removed |
|---|---|---|
| CN | Later diagnosed MCI or AD | 215 |
| CN | No 3 T scan within 90 days of baseline | 431 |
| CN | Missing age or sex | 0 |
| MCI | Enrolled in ADNI4 | 234 |
| MCI | Reverted to CN | 114 |
| MCI | Follow-up under 36 months without conversion | 486 |
| MCI | No 3 T scan within 90 days of baseline | 262 |
| MCI | Missing age or sex | 0 |
| CN | Failed scan QC (misaligned or artefact, no usable same-session T1) | 7 |
| MCI | Failed scan QC (misaligned, no other T1 in the session) | 1 |
