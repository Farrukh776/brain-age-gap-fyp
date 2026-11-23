# Session Log

One entry per working session, newest at the top. Read the top entry first when starting a session.
Step numbers refer to `ROADMAP.md`. Never put subject IDs or data values here.

---

## 2026-11-23 (Session 7)

**Current step:** 6 (brain-age model). Features extracted, SVR / ridge baseline trained, 3D CNN
data loader and training script built and tested on the laptop CPU. Next is the full CNN
training run, which needs the lab GPU.

**Done**
- Decision 22: baseline features are 4 mm voxels (4 x 4 x 4 block means of each preprocessed
  scan, 29,520 blocks inside the template brain mask). New `src/brain_age/extract_features.py`
  writes one matrix for all 2,489 scans (CN train 1,755 / val 140 / test 141, MCI 316 / 68 / 69)
  in about 3 minutes. No statistics are learned at this stage.
- New `src/brain_age/train_baseline.py`: standardise -> PCA -> ridge or SVR (RBF), tuned with
  5-fold CV on all 1,755 training scans, folds grouped by subject; PCA fitted inside each fold.
  Test set not loaded.
- Results (validation, 140 CN baselines): ridge MAE 3.79, RMSE 4.88, r 0.71 (CV MAE 4.04);
  SVR MAE 4.25 (CV 4.70); always guessing the mean training age gives MAE 6.15. Ridge was best
  at the largest PCA size, so the grid was extended to 1,200 components; it flattened (decision 24).
- Gap-vs-age slope about -0.47: old subjects predicted too young, young too old. Expected;
  bias correction is a later roadmap item. Out-of-fold training predictions are saved for it.
- C: ran out of space while saving models. `data/features` and `outputs/brain_age` moved to
  `D:\fyp-data` with folder links (decision 23). C: has only about 0.2 GB free.
- New `src/brain_age/data.py`: scan table with ages for all 2,489 scans and a PyTorch dataset
  (1 mm -> 2 mm block mean, 80 x 96 x 80; training-only left-right flip and shift up to 2
  voxels). Tested on 20 training + 5 validation scans: shapes, values, ages, each scan once per
  epoch, validation unchanged between reads; picture checked by eye (`loader_check.png`, local).
  Loading takes about 0.6 s per scan on the laptop (2 workers), so one epoch would take about
  17 minutes from the .nii.gz files.
- Found while testing: 3 MCI scans (one per split) replaced by another T1 of the same session
  (decision 17) were missing from the age lookup; they now take the age of the image they
  replaced. CN scans were not affected.
- New `src/brain_age/cnn.py` (compact 3D CNN, 1,176,721 weights) and `src/brain_age/train_cnn.py`
  (L1 loss, AdamW, lr halving and early stopping on validation MAE, best epoch kept, resume from
  `last.pt`, `--smoke` test mode). Decision 25. Optional 2 mm cache in `data.py`
  (`cnn.cache_dir`, off by default).
- Smoke tests on the laptop CPU (12 train / 8 val scans, 2 epochs): runs end to end; `--resume`
  of a finished run gives the same result; runs with and without the cache give identical
  numbers (reproducible with seed 42). Test outputs deleted. On the CPU about 0.6 s per scan per
  epoch, so a full run (up to 100 epochs x 1,755 scans) is not practical on the laptop.

**Next (session 8)**
1. Supervisor review of `cohort_definition.md` (still pending).
2. Free space on C: (outside the project) before larger runs.
3. Confirm lab GPU access (GPU model, memory, booking); set up the `fyp` env there with the CUDA
   build of PyTorch; copy the processed scans and re-run `verify_processed.py` (decision 14:
   no personal cloud).
4. On the GPU: set `cnn.cache_dir`, run `train_cnn --smoke`, then the full run `--run cnn-1`.
   Stop it after a few epochs once and check `--resume` continues correctly.
5. Log the CNN run in `experiment_log.md` and compare with ridge (val MAE 3.79).

**Open questions**
- SVR with an RBF kernel did much worse than ridge; a linear kernel or tuned gamma was not
  tried. Ridge is enough for a baseline, revisit only if the CNN comparison needs it.

---

## 2026-11-22 (Session 6)

**Current step:** 6 (brain-age model). Preprocessing was re-checked end to end; the data is now
final. Feature extraction has not started.

**Done**
- FreeSurfer coverage check (`src/brain_age/check_fs_coverage.py`): the UCSFFSX7 table misses
  3 to 8% of each split and is mostly unrated. Decision 19: the baseline uses features from our
  own preprocessed images.
- New `src/preprocessing/verify_processed.py`: every scan the models read passed file, header,
  value and z-score checks; tables and splits consistent.
- Found 4 people enrolled under two subject IDs (3 split across train/val/test). Decision 20:
  keep the earlier enrolment. Removed 4 CN subjects (train 2, val 1, test 1).
- Found tilted scans the eye check had missed. New `src/preprocessing/check_tilt.py` (study
  mean image, rotation per scan; decision 21). 3 + 18 scans re-aligned with `--retry-wide` and
  checked by eye; 15 retries that got worse were restored; 1 CN training baseline still
  misaligned and removed (decision 17).
- Final: 1,388 subjects, 2,489 scans. CN train 654 subjects / 1,755 scans, val 140, test 141;
  MCI 316 / 68 / 69 (unchanged). Counts in `cohort_definition.md`, method in
  `preprocessing_log.md`.
- Report figure of the MNI ICBM152 2009a template with brain mask and crop box:
  `docs/figures/mni_template.png` (`src/preprocessing/template_figure.py`).

**Next (session 7)**
1. Supervisor review of `cohort_definition.md` (still pending; now includes decision 20).
2. Choose the image features for the SVR / ridge baseline (for example atlas regional means or
   downsampled voxels + PCA fitted on train), then extract them for all CN scans.
3. Train the SVR / ridge baseline with subject-grouped CV on train; report MAE, RMSE and
   correlation on validation.

**Open questions**
- 33 scans stay at 4 to 6 degrees against the study mean after retry; checked by eye and
  aligned (head shape). If the baseline shows outliers among them, revisit.

---

## 2026-11-18 (Session 5)

**Current step:** 5 (MRI preprocessing) is complete, including the later CN training visits.
Next is step 6 (brain-age model).

**Done**
- The later-visit download (1,110 scans, two zips) was unpacked into its own folders and
  checked complete (`check_scans.py --scans data/tables/later_visit_scans.csv`, new option).
- Preprocessed all 1,110 with 3 workers. The laptop lost power at 1,068; the run was restarted
  and skipped finished scans. 0 errors.
- Every scan checked by eye: 1,079 ok, 28 misaligned, 3 failed (artefact or unusable image).
  `--retry` fixed 29 of 36 problem scans; `--retry-wide` fixed none of the last 6.
- 9 later visits failed QC and were dropped; no subject was removed.
- New `src/preprocessing/make_training_scans.py` writes `data/tables/cn_train_scans.csv`
  (local only): 657 training subjects, 657 baseline + 1,101 later visits = 1,758 scans.
  All 1,101 later outputs checked: 160 x 192 x 160 at 1 mm, no blank or invalid values.
- Final data summary (subjects vs scans per cohort and split) added to
  `cohort_definition.md`; decision 18 records how failed later visits are handled.
- Steps 1 to 5 are done except the items that need teammates or the supervisor (IDA access
  for all members, collaborators, env on their machines, GPU booking, supervisor review).

**Next (session 6)**
1. Supervisor review of `cohort_definition.md` (still pending from step 4).
2. Step 6: the brain-age model reads CN training scans from `data/tables/cn_train_scans.csv`
   and validation/test (and MCI) scans from `data/tables/scan_qc_final.csv`
   (status not `excluded`).
3. Decide whether the SVR / ridge baseline uses the FreeSurfer 7.x table or features from the
   preprocessed images, and whether it trains on baseline scans only or all 1,758.
4. Start step 6 with features for the CN subjects and the SVR / ridge baseline.

**Open questions**
- Five later visits came out with the brain smaller than the template outline after every
  retry (scalp matched to the brain edge). They are dropped; if many more scans of this kind
  appear later, revisit the registration (for example a brain-only template).

---

## 2026-11-16 (Session 4)

**Current step:** 5 (MRI preprocessing) is finished for the baseline scans. Next is step 6
(brain-age model).

**Done**
- All 1,401 essential baseline scans downloaded and checked complete (`check_scans.py`).
  Two input fixes: multi-frame single-file DICOMs are read directly; non-T1 images (localizer,
  FLAIR) are never chosen (decision 15).
- Full preprocessing run with 3 workers (about 7 h): 1,401 succeeded, 0 errors.
- Every scan checked by eye on contact sheets (`qc_sheets.py`, `record_review.py`; decision 16).
  The 0.30 score alone missed slight tilts and flagged some aligned scans.
- Misaligned scans redone with `--retry` (58 of 65 fixed) and `--retry-wide` (0 of 7 fixed).
- Image-quality rule set (decision 17): artefacts over brain tissue fail QC.
- Failed scans replaced by another T1 from the same session where one existed (6 downloaded,
  3 passed). Final: 1,393 of 1,401 subjects usable; CN 940 (657/141/142), MCI 453 (316/68/69,
  1 progressor removed from train). Counts in `cohort_definition.md`, method and failures in
  `preprocessing_log.md`. Per-subject table (local only): `data/tables/scan_qc_final.csv`.

- Checked the cohort files again: `cohort_definition.md` (section A) trains the brain-age
  model on all CN visits of training subjects; validation and test use baseline only. That is
  1,110 later-visit scans from 335 of the 657 CN training subjects (1,114 minus 4 of subjects
  removed at QC). The session 3 figure of about 1,550 also counted validation and test visits,
  which are never used. Training set: 657 baseline + 1,110 later = 1,767 scans.
- `make_download_lists.py --later-visits` wrote `data/tables/later_visit_scans.csv` and 6 lists
  (`download_lists/later_01.txt` to `later_06.txt`, local only). Farrukh started the download
  on 2026-11-16 (into `D:yp-data
aw\scans`).

**Next (session 5)**
1. Ask Farrukh whether the later-visit download has finished. Unpack each zip into its own new
   folder under `data/raw/scans`, then run `check_scans.py` style completeness checks on the
   1,110 IDs.
2. Preprocess them: `python -m src.preprocessing.preprocess --scans
   data/tables/later_visit_scans.csv --workers 3` (about 6 h). Check every scan by eye
   (`qc_sheets.py --new-only`, `record_review.py`), `--retry` the misaligned ones, apply
   decision 17. A failed later visit is simply dropped (the subject keeps its other scans);
   record the counts in `preprocessing_log.md` and add the visits to `scan_qc_final.csv` or a
   companion table.
3. These later visits are needed for the 3D CNN. Step 6 can start without them: features for
   the CN subjects and the SVR / ridge baseline on baseline scans. Decide first whether to use
   the FreeSurfer 7.x table for the baseline model, and whether the baseline model also uses
   the later visits.
4. Every later code must read the image to use from `data/tables/scan_qc_final.csv` (column
   `image_id`, rows with status `excluded` dropped), not from the cohort tables.
5. Supervisor review of `cohort_definition.md` (still pending from step 4).

**Open questions**
- Does the SVR / ridge baseline train on baseline scans only or on all training visits?
  (The 3D CNN uses all training visits, per `cohort_definition.md`.)

---

## 2026-11-14 to 2026-11-15 (Session 3)

**Current step:** 5 (MRI preprocessing). Pilot done; the download of the essential scans is in
progress. Step 4 is finished except supervisor review. Steps 2 and 3 still have open items
that need teammates.

**Done**
- Downloaded the ADNI tables into `data/raw/tables` (diagnosis, demographics, MMSE, registry,
  data dictionary, 1.5 T and 3 T scan information, T1 image manifest, FreeSurfer 7.x).
- Decided the cohort rules (decisions 7 to 12): 3 T only, 36-month window, exclusions, scan
  matching. `docs/cohort_definition.md` is filled in with counts and an exclusion log.
- Wrote `src/cohort/inspect_tables.py`, `build_cohorts.py` and `make_splits.py` (counts only).
- Built both cohort tables and the 70/15/15 subject-level splits; no overlaps.
- Step 5 pilot done: `src/preprocessing/select_pilot.py` picked 10 scans (5 CN, 5 MCI, training
  subjects, mixed phases and scan types); they were downloaded through an ADNI collection
  (Advanced Image Search, paste image IDs, 1-Click Download) and unpacked into `data/raw/scans`.
- Wrote `src/preprocessing/preprocess.py` (decision 13). All 10 pilot scans aligned correctly,
  about 24 s per scan, 12.5 MB per output. Settings and the registration problems found on the
  way are in `docs/preprocessing_log.md`.

- `src/preprocessing/make_download_lists.py` wrote 7 lists of image IDs (200 each, 1,391
  essential scans: 454 MCI + 947 CN baseline, minus the 10 pilot scans) to
  `data/tables/download_lists/`. Farrukh pasted them into Advanced Image Search and added all
  of them to one ADNI collection named `essential`.
- Download state when the session ended (2026-11-15, afternoon): 1-Click Download of the whole
  collection crashed the site twice. One large file was then downloading into
  `data/raw/scans` (about 3.7 GB so far, still a `.crdownload`). How that download was started
  (Advanced Download, or a smaller selection) is not recorded; ask Farrukh.

**Next (start of session 4, once the downloads have finished)**
1. Ask Farrukh how the download went and which method worked. Then list `data/raw/scans`:
   finished files end in `.zip`, unfinished ones in `.crdownload`. Test every zip
   (`unzip -tq`).
2. Unpack each zip into its own folder under `data/raw/scans` (the pilot is in `pilot10/`).
   Scan folders are named `I<image_id>`; `preprocess.py` finds them anywhere under
   `data/raw/scans`.
3. Re-run `python -m src.preprocessing.make_download_lists`. It reports how many essential
   scans are still missing and rewrites the lists with only those. Expected total: 1,401.
   If scans are missing, download them in smaller collections (one per list; reduce
   `BATCH_SIZE` in the script if 200 is still too many for the site).
4. Add parallel processing to `preprocess.py` (several scans at once; memory is about 6 GB, so
   start with 3 or 4 workers), then run the full batch on a scan list made from the cohort
   tables (MCI scans and CN baseline scans). About 24 s per scan on one core; the script
   skips scans already done, so it can be stopped and restarted.
5. Review flagged scans (score below 0.30) and a random sample of the rest on their QC
   pictures in `outputs/preprocessing/qc`; log the run and any failures in
   `docs/preprocessing_log.md`; remove failed subjects from the cohort (never move anyone
   between splits). Tick the remaining step 5 items in `ROADMAP.md`.
6. Then step 6 (brain-age model). Decide first whether to use the FreeSurfer 7.x table for
   the baseline model.

**Also pending**
- Show `docs/cohort_definition.md` to the supervisor (rules were decided without his review).
- Extra CN visits (about 1,550 scans) are downloaded later, for CNN training only.
- Laptop settings changed for the overnight download: sleep when plugged in was to be set to
  Never, and Chrome's download folder to `D:\fyp-data\raw\scans`. Set sleep back afterwards.
- Do not screenshot ADNI result pages into the chat (they list subject IDs); describe them.
- Steps 2 and 3: add teammates as collaborators; they set up the env and download the same
  table versions with their own ADNI accounts.

**Open questions**
- 3D CNN training: a university lab GPU is available (used in person). Still to confirm:
  GPU model and memory, storage, whether packages can be installed, booking or time limits.
  Colab is avoided because uploading ADNI scans may breach the Data Use Agreement.
- D: has 318 GB free, enough for all scans. Laptop: 8 cores, about 6 GB memory.
- Use ADNI's FreeSurfer 7.x table for regional volumes (baseline model, E1)? Decide in step 6.
- The progression test set is small (69 subjects, 24 progressors); expect wide intervals.
- Teammates must rebuild the same splits: the split files are local only, so check that the
  scripts give identical counts on their machines.

---

## 2026-11-13 (Session 2)

**Current step:** 3 (Environment); step 2 still has open items that need teammates.

**Done**
- Pushed `CLAUDE.md`. Repo kept public (decision 3). Streams shared by all members (decision 4).
- Installed Miniconda on D: and created the `fyp` env (Python 3.11, CPU-only PyTorch) on
  Farrukh's laptop (decision 5). All packages in `requirements.txt` import; `pip check` is clean.
- Scan storage set up on D: with folder links from `data/` (decision 6).

**Next**
- Step 2: add teammates and supervisor as collaborators; each member pushes one test branch.
- Step 3: teammates set up the `fyp` env; confirm GPU access.
- Then step 4 (download clinical tables, fill `cohort_definition.md`).

**Open questions**
- Where will the 3D CNN be trained? Farrukh's laptop has no NVIDIA GPU and 6 GB RAM.
  Options: a teammate's GPU, a university lab, or Colab/Kaggle if the ADNI DUA allows uploads.
- FYP-I evaluation date?
- Do all three members have their own ADNI login?

---

## 2026-11-13 (Session 1)

**Current step:** 2 (Repository)

**Done**
- Read the proposal; agreed FYP-I covers everything up to and including E2, in pipeline order.
- ADNI access granted.
- Created the starter repo files: README, .gitignore, requirements, configs, docs, src skeleton.

**Next**
- ~~Create the private GitHub repo and push the starter files.~~ Done; repo kept public (decision 3).
- Add teammates and supervisor as collaborators.
- ~~Fill the stream column (who does what) in `README.md`.~~ Done; streams are shared (decision 4).
- Then step 3 (environment) and step 4 (download clinical tables, fill `cohort_definition.md`).

**Open questions**
- FYP-I evaluation date?
- Do all three members have their own ADNI login?
- GPU and storage for scans?

---

<!-- Template: copy for each new session

## YYYY-MM-DD (Session N)

**Current step:**

**Done**
-

**Next**
-

**Open questions**
-

-->
