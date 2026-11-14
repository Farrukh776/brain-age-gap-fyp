# Decision Log

Record every choice that affects results, with the reason. Newest at the bottom.

| # | Date | Decision | Reason | Decided by |
|---|---|---|---|---|
| 1 | 2026-11-13 | FYP-I covers everything up to and including E2, done in pipeline order (preprocessing, brain age, BAG, E1, E2) | FYP phase requirement | Team |
| 2 | 2026-11-13 | ADNI data and subject IDs are never committed to git | ADNI Data Use Agreement | Team |
| 3 | 2026-11-13 | The GitHub repo stays public (code, configs and docs only) | Farrukh's preference; the repo holds no ADNI data, and decision 2 plus `.gitignore` keep it that way | Farrukh |
| 4 | 2026-11-13 | No fixed stream per member; all three work on every stream together | Team preference | Farrukh |
| 5 | 2026-11-13 | On Farrukh's laptop, Miniconda and the `fyp` env live on D: (`D:\miniconda3`), Python from conda-forge, CPU-only PyTorch | C: has about 6 GB free and the laptop has no NVIDIA GPU; 3D CNN training will need another machine | Farrukh |
| 6 | 2026-11-13 | On Farrukh's laptop, scans live in `D:\fyp-data`; `data/raw`, `data/bids` and `data/processed` are folder links to it, `data/tables` stays in the repo folder. Config paths stay relative (`data/...`) | C: is nearly full; relative paths keep the same configs working on every member's machine | Farrukh (delegated to Claude) |
| 7 | 2026-11-14 | 3 T scans only, for both cohorts | 43% of MCI progressors but 10% of CN subjects were scanned at 1.5 T, so BAG could reflect the scanner; 3 T only needs no correction step. Cost: about 148 progressors instead of 261 | Farrukh (delegated to Claude) |
| 8 | 2026-11-14 | Brain-age cohort uses ADNI1, GO, 2, 3 and 4; progression cohort uses ADNI1, GO, 2 and 3 | ADNI4 adds about 320 CN subjects, but its MCI subjects enrolled too recently for a 36-month outcome and would enter almost only as progressors | Farrukh (delegated to Claude) |
| 9 | 2026-11-14 | Progression outcome: dementia within 36 months = progressor; no dementia and follow-up of 36 months or more = stable; MCI subjects who revert to CN are excluded | 36 months is common in the cited literature and keeps more progressors than a shorter window; reverters have uncertain labels | Farrukh (delegated to Claude) |
| 10 | 2026-11-14 | CN subjects later diagnosed MCI or AD are excluded from the brain-age cohort | The model should learn normal ageing; early undetected disease would shrink BAG in MCI subjects | Farrukh (delegated to Claude) |
| 11 | 2026-11-14 | A scan matches the baseline diagnosis if acquired within 90 days of it; nearest scan wins | Simple rule that covers the usual gap between screening scan and baseline visit | Farrukh (delegated to Claude) |
| 12 | 2026-11-14 | ADNIMERGE is not used; cohorts are built from the raw tables (DXSUM, PTDEMOG, MMSE, REGISTRY, scan information, T1 manifest) | ADNIMERGE is now an R package only; the raw tables hold the same information | Farrukh (delegated to Claude) |
