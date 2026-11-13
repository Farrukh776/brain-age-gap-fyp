# Cohort Definition

**Status:** DRAFT. Must be finalised and agreed with the supervisor before any model is trained.
Items marked TODO are decisions the team still has to make.

## Data source
- Dataset: ADNI. Phases included: TODO (e.g. ADNI1 / GO / 2 / 3)
- Tables used: TODO (list exact file names and download dates)
- Imaging: T1-weighted structural MRI, baseline visit. Field strength: TODO

## A. Brain-age cohort (cognitively unimpaired)
Inclusion:
- Diagnosis CN at the scan visit
- Usable T1 scan that passes QC
- Known chronological age at scan

Exclusion:
- TODO (e.g. later conversion to MCI/AD: include or exclude?)

Scans per subject: TODO (baseline only, or multiple visits with subject-level split)

## B. Progression cohort (MCI)
Inclusion:
- Diagnosis MCI at baseline
- Usable baseline T1 scan that passes QC

Outcome definition:
- Follow-up window: TODO (proposed: 36 months)
- **Progressor (label 1):** diagnosis changes to AD within the window
- **Stable (label 0):** remains MCI, with follow-up at least as long as the window

Exclusion:
- Reverts to CN during follow-up: TODO (proposed: exclude)
- Follow-up shorter than the window without conversion
- Missing age or failed scan QC

## Visit alignment
- How a scan is matched to a diagnosis visit: TODO (e.g. same visit code, max days apart)

## Splits
- Subject-level, stratified by label. Ratio 70 / 15 / 15. Seed: 42
- Split files are created once and never regenerated

## Counts (fill after building)
| Cohort | Start | After exclusions | Train | Val | Test |
|---|---|---|---|---|---|
| CN | | | | | |
| MCI stable | | | | | |
| MCI progressor | | | | | |

## Exclusion log
| Rule | Subjects removed |
|---|---|
| | |
