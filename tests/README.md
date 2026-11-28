Unit tests. Run from the repo root: `python -m pytest` (about 20 seconds).

- `test_configs.py`: E1 and E2 differ only by the age input; no BAG or clinical inputs in FYP-I;
  seed 42 everywhere; frozen model hashes present.
- `test_metrics.py`: DeLong test against scikit-learn and a bootstrap; cut-off and threshold
  metrics checked by hand (simulated data, no ADNI data needed).
- `test_data_integrity.py`: no subject in two splits or both cohorts; CN training scans only from
  training subjects; MCI features, BAG table and splits match; E1 and E2 used the same subjects and
  folds; MCI test subjects never in training outputs; the test-set lock refuses a second run.
  These read local ADNI-derived files and skip themselves when the data is not on the machine.
