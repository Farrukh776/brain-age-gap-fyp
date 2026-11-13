# Decision Log

Record every choice that affects results, with the reason. Newest at the bottom.

| # | Date | Decision | Reason | Decided by |
|---|---|---|---|---|
| 1 | 2026-11-13 | FYP-I covers everything up to and including E2, done in pipeline order (preprocessing, brain age, BAG, E1, E2) | FYP phase requirement | Team |
| 2 | 2026-11-13 | ADNI data and subject IDs are never committed to git | ADNI Data Use Agreement | Team |
| 3 | 2026-11-13 | The GitHub repo stays public (code, configs and docs only) | Farrukh's preference; the repo holds no ADNI data, and decision 2 plus `.gitignore` keep it that way | Farrukh |
| 4 | 2026-11-13 | No fixed stream per member; all three work on every stream together | Team preference | Farrukh |
| 5 | 2026-11-13 | On Farrukh's laptop, Miniconda and the `fyp` env live on D: (`D:\miniconda3`), Python from conda-forge, CPU-only PyTorch | C: has about 6 GB free and the laptop has no NVIDIA GPU; 3D CNN training will need another machine | Farrukh |
