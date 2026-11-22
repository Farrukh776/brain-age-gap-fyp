# Preprocessing Log

## Pipeline
- Tool and version: SimpleITK 2.5.7 and nilearn 0.14.1 (`src/preprocessing/preprocess.py`)
- Input: original T1 DICOM series from ADNI, one per scan session (see `cohort_definition.md`)
- Steps (in order):
  1. Read the DICOM series and reorient to LPS
  2. N4 bias field correction (field estimated on a 4x shrunken copy, Otsu head mask)
  3. Rigid then affine registration to the template (no non-linear warping, so atrophy is kept)
  4. Resample to the template grid, linear interpolation, 1 mm voxels
  5. Crop to a fixed box centred on the template brain
  6. Clip intensities to the 0.5 to 99.5 percentiles and z-score, both inside the template brain mask
  7. Save as `<image_id>.nii.gz` with a QC picture and a QC score
- Template / output space: MNI ICBM152 2009a non-linear symmetric, 1 mm, with skull
  (downloaded by nilearn `fetch_icbm152_2009`)
- Output shape and voxel size: 160 x 192 x 160 at 1 mm (down-sampled to 2 mm when loaded by the CNN)
- Skull stripping: no. The skull is partly cut by the crop box and the rest is left in.
  Revisit only if the brain-age model appears to rely on non-brain tissue.
- Command or config used: `configs/preprocessing.yaml`;
  `python -m src.preprocessing.preprocess --scans <csv with image_id column>`

## Registration settings found during the pilot
- The match is scored only inside the template brain mask, shrunk by 3 mm. When skull or scalp
  took part, the bright scalp was matched to the brain edge and heads came out about 10% small.
  This happened with both the brain-only and the with-skull template.
- Similarity measure: correlation. Mattes mutual information aligned some pilot scans and
  failed on others.
- Starting position: centre of mass of the top 180 mm of the head, so the neck is ignored.
- Optimiser: regular step gradient descent, 25% random sampling (seed 42); rigid at shrink
  factors 8 and 4, then affine at 4 and 2.

## Input formats and running in parallel (2026-11-16)
- Most scans are one DICOM file per slice (154 to 211 slices). 27 essential scans (ADNI2,
  ADNIGO, ADNI3) store the whole volume in one multi-frame file (about 22 MB); these are read
  directly, because the series reader returned them as 4D images and they failed.
- `src/preprocessing/check_scans.py` checks every essential scan is on disk with no empty or
  cut-off files (counts only; results in `outputs/preprocessing/scan_check.csv`).
- `--workers N` processes N scans at once. On the 6 GB laptop: 3 workers, about 0.5 GB memory
  each, about 50 s per scan per worker, so about 18 s per scan overall (24 s with one worker).
  More workers run out of memory.

## QC criteria (defined before the full run)
- Automatic: correlation with the template inside the brain mask. Below 0.30 the scan is flagged.
  (Pilot: well-aligned scans scored 0.40 to 0.57; misaligned ones below 0.20.)
- Visual: every flagged scan, plus a random sample of unflagged scans, is checked on its QC
  picture: brain fills the red template outline, midline is vertical, nothing is cut off.
  Changed during the full run (decision 16): **every** scan is checked by eye, because the
  score missed slight tilts (0.38 to 0.47) and flagged some aligned scans (weak contrast,
  strong atrophy).
- Image quality (decision 17): motion, ringing, streak or noise artefacts over brain tissue
  fail QC even when aligned; artefacts outside the brain only are accepted.
- A scan that fails is replaced by another T1 from the same session if one exists and passes;
  otherwise it is logged below and its subject is removed from the cohort and its split.

## Visual review tools (2026-11-16)
- `src/preprocessing/qc_sheets.py` pastes QC pictures onto contact sheets of 10, lowest score
  first (`--new-only` leaves out scans already reviewed).
- `src/preprocessing/record_review.py` stores verdicts per sheet in
  `outputs/preprocessing/visual_review.csv` (local only).
- `src/preprocessing/view_scan.py` shows one scan raw and preprocessed side by side.
- Final per-subject table (local only): `data/tables/scan_qc_final.csv` (which image each
  subject uses, or excluded, with the reason).

## Retry registration (2026-11-16)
- `--retry`: 9 rigid starts (pitch 0, +-15, +-30 degrees; height 0, +-20, +-40 mm), the start
  with the best correlation is refined to affine; an affine that stretches any axis outside
  0.75 to 1.35 is rejected and the rigid result kept. About 2 min per scan with 3 workers.
- `--retry-wide`: also the 23 other 90-degree orientations (32 starts, about 6 min per scan).
- Previous outputs of retried scans were backed up locally before each retry.

## Later CN training visits (2026-11-16)
- The brain-age training set uses every CN visit of the training subjects
  (`cohort_definition.md`, section A): 1,110 later-visit scans on top of the baseline scans.
- `python -m src.preprocessing.make_download_lists --later-visits` writes
  `data/tables/later_visit_scans.csv` (the input for `preprocess.py`) and the ADNI lists.
- Same pipeline, same visual check and same rules (decisions 16, 17). A failed later visit is
  dropped; the subject keeps its other scans and stays in its split.
- Result (2026-11-18): all 1,110 downloaded and complete (`check_scans.py --scans`);
  1,101 passed the visual check, 9 failed (see Failures). Training set after QC: 657 subjects,
  657 baseline + 1,101 later visits = 1,758 scans, written by
  `python -m src.preprocessing.make_training_scans` to `data/tables/cn_train_scans.csv`
  (local only; the brain-age model reads its scan list from there).

## Final check before modelling (2026-11-22)
`python -m src.preprocessing.verify_processed --workers 6` (about 6 min) re-reads every scan the
models use and checks: file decompresses; shape, voxel size, orientation and origin equal the
template box; float32; no NaN/inf; z-score inside the brain mask (mean 0, SD 1); share of brain at
the clip limits (about 0.5% expected); top of brain not cut off; correlation equals the QC record;
identical files; same person under two subject IDs; tables and splits consistent. FLAG lines are
errors, LOOK lines are scans to view by eye. Re-run it after copying scans to another machine.

All 2,494 scans passed the file, header and value checks. Two problems were found:
1. **Tilted scans the eye check missed.** Each CN training scan was registered to the same
   subject's most typical other scan: normal scans need under 2.1 mm and 0.3 degrees of
   movement, but about 15 needed 5 to 12 degrees. To check scans without a second visit, a
   study mean image was built from all scans (in MNI space) and every scan registered to it
   (affine). Rotation separates the two groups (controls up to 2.5 degrees, confirmed tilts
   3.7 to 14 degrees); millimetres of movement do not (affine also removes normal head-size
   differences); this check is `python -m src.preprocessing.check_tilt --workers 6` (about
   70 min). 66 scans rotated more than 4 degrees and were redone with `--retry-wide`
   (originals backed up): 18 were tilted and are now 0.3 to 2.8 degrees (all checked by eye);
   33 came back unchanged (head shape, not misalignment; the 6 above 5 degrees were checked by
   eye and are aligned); 15 got worse and the originals were restored. Also 4 scans found
   earlier in this check: 3 fixed with `--retry-wide`, 1 CN training baseline still too small
   after every retry and removed (decision 17).
2. **Same person under two subject IDs** (4 pairs, decision 20); found because their images
   correlated 0.92 to 0.98, against at most 0.82 for different people of the same site, birth
   month and sex.

The template used for registration is shown in `docs/figures/mni_template.png`
(`python -m src.preprocessing.template_figure`).

## Run history
| Date | Who | Scans in | Succeeded | Failed | Notes |
|---|---|---|---|---|---|
| 2026-11-14 | Farrukh (with Claude) | 10 (pilot: 5 CN, 5 MCI, mixed phases and scanner types) | 10 | 0 | About 24 s per scan on the laptop. All 10 checked by eye, all aligned. |
| 2026-11-16 | Farrukh (with Claude) | 10 (test of 3 workers on 6 scans, 2 single-file scans, the 2 replacement images from decision 15) | 10 | 0 | 1 flagged (0.28); checked by eye and aligned, low score from weak contrast. The 2 single-file scans first failed (4D read), passed after the fix. All 10 checked by eye. |
| 2026-11-16 | Farrukh (with Claude) | 1,391 (full run, all remaining essential scans; 1,401 in total with the 10 pilot scans) | 1,401 | 0 | 3 workers, about 18 s per scan overall, about 7 h. 52 flagged below 0.30. All 1,401 checked by eye: 1,341 ok, 55 misaligned, 5 image artefacts. |
| 2026-11-16 | Farrukh (with Claude) | 65 (`--retry`: the 55 misaligned plus 10 more scoring below 0.30) | 65 | 0 | 58 aligned after the retry, checked by eye; 7 still misaligned. |
| 2026-11-16 | Farrukh (with Claude) | 7 (`--retry-wide`) | 7 | 0 | None aligned (heads with the face and neck filling the brain box). |
| 2026-11-18 | Farrukh (with Claude) | 1,110 (later CN training visits) | 1,110 | 0 | 3 workers. The laptop lost power at 1,068; the run was restarted and skipped finished scans. All 1,110 checked by eye: 1,079 ok, 28 misaligned, 3 failed (artefact or unusable image). |
| 2026-11-18 | Farrukh (with Claude) | 36 (`--retry`: 28 misaligned plus 8 more scoring below 0.30) | 36 | 0 | 29 aligned after the retry, checked by eye; 6 still misaligned (5 with the brain smaller than the outline, 1 pitched); the unusable image stayed failed. |
| 2026-11-18 | Farrukh (with Claude) | 6 (`--retry-wide`) | 6 | 0 | None aligned. |
| 2026-11-22 | Farrukh (with Claude) | 3 (`--retry-wide`, misaligned scans found in the final check) | 3 | 0 | All 3 aligned, checked by eye. 1 more scan (CN train baseline) still too small after `--retry-wide`; subject removed. |
| 2026-11-22 | Farrukh (with Claude) | 66 (`--retry-wide`, rotated more than 4 degrees against the study mean) | 66 | 0 | 6 workers, about 1.5 h. 18 fixed (checked by eye), 33 unchanged, 15 worse and restored from backup. |
| 2026-11-16 | Farrukh (with Claude) | 6 (other T1 images from the same session, downloaded for failed scans) | 6 | 0 | Checked by eye: 3 passed and replace a failed scan; 3 failed (1 misaligned, 1 ringing, 1 noisy backup that was not needed). See Failures. |

## Failures
Counts only; image IDs are in the local tables named above.

| Reason | Scans | Action | Result |
|---|---|---|---|
| Misaligned after `--retry` and `--retry-wide` | 7 (6 CN, 1 MCI) | Same-session T1 used where one exists (2 of 7) | 1 MCI replaced; 1 CN repeat also misaligned; 6 subjects excluded (5 CN, 1 MCI) |
| Artefact over brain tissue (motion, ringing, streaks, noise) | 4 (2 CN, 2 MCI) | Same-session T1 used where one exists (3 of 4) | 2 MCI replaced; 1 CN repeat had the same ringing; 2 CN subjects excluded |
| Line artefact outside the brain only | 1 | Kept (decision 17) | ok |
| Later CN visit: artefact or unusable image | 3 | Visit dropped | subjects keep their other scans |
| Later CN visit: misaligned after `--retry` and `--retry-wide` | 6 | Visit dropped | subjects keep their other scans |

| Baseline misaligned, found in the final check (2026-11-22) | 1 CN | No other T1 in the session | subject excluded |

Final (2026-11-22): 1,392 of 1,401 subjects have a usable scan; 4 more CN subjects were removed
as duplicate enrolments (decision 20), leaving 1,388. CN 935 (train 654, val 140, test 141);
MCI 453 of 454 (train 316, val 68, test 69; 1 progressor excluded from train).
