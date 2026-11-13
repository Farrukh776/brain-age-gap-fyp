# data/ (local only)

Everything here except this file is ignored by git. Do not change that.

Suggested layout:

```
data/
  raw/        ADNI downloads exactly as received (scans, clinical tables)
  bids/       Scans converted to BIDS, if a converter is used
  processed/  Preprocessed scans
  tables/     Cohort tables, split files, feature tables, BAG table
```

Each member downloads data with their own ADNI account. Do not share ADNI files through
GitHub, public drives or messaging apps.
