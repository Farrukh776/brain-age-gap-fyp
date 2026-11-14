# Session Log

One entry per working session, newest at the top. Read the top entry first when starting a session.
Step numbers refer to `ROADMAP.md`. Never put subject IDs or data values here.

---

## 2026-11-14 (Session 3)

**Current step:** 4 (Cohort and splits) finished except supervisor review; step 5 is next.
Steps 2 and 3 still have open items that need teammates.

**Done**
- Downloaded the ADNI tables into `data/raw/tables` (diagnosis, demographics, MMSE, registry,
  data dictionary, 1.5 T and 3 T scan information, T1 image manifest, FreeSurfer 7.x).
- Decided the cohort rules (decisions 7 to 12): 3 T only, 36-month window, exclusions, scan
  matching. `docs/cohort_definition.md` is filled in with counts and an exclusion log.
- Wrote `src/cohort/inspect_tables.py`, `build_cohorts.py` and `make_splits.py` (counts only).
- Built both cohort tables and the 70/15/15 subject-level splits; no overlaps.

**Next**
- Show `docs/cohort_definition.md` to the supervisor (rules were decided without his review).
- Step 5: choose the preprocessing tool, then download the baseline T1 scans for a 10-scan
  pilot. Use original images from the "Clinical T1w Imaging Cohort" (ARC Builder > Downloads >
  Images); do not download the full 258 GB set.
- Steps 2 and 3: add teammates as collaborators; they set up the env and download the same
  table versions with their own ADNI accounts.

**Open questions**
- Where will the 3D CNN be trained? No NVIDIA GPU on Farrukh's laptop.
- How much free space is on D: for scans (about 950 CN + 454 MCI baseline scans, more with
  extra CN visits)?
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
