# Session Log

One entry per working session, newest at the top. Read the top entry first when starting a session.
Step numbers refer to `ROADMAP.md`. Never put subject IDs or data values here.

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
