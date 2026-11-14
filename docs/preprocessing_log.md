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

## QC criteria (defined before the full run)
- Automatic: correlation with the template inside the brain mask. Below 0.30 the scan is flagged.
  (Pilot: well-aligned scans scored 0.40 to 0.57; misaligned ones below 0.20.)
- Visual: every flagged scan, plus a random sample of unflagged scans, is checked on its QC
  picture: brain fills the red template outline, midline is vertical, nothing is cut off.
- A scan that fails is logged below and its subject is removed from the cohort and its split.

## Run history
| Date | Who | Scans in | Succeeded | Failed | Notes |
|---|---|---|---|---|---|
| 2026-11-14 | Farrukh (with Claude) | 10 (pilot: 5 CN, 5 MCI, mixed phases and scanner types) | 10 | 0 | About 24 s per scan on the laptop. All 10 checked by eye, all aligned. |

## Failures
| Scan ID | Stage | Reason | Action |
|---|---|---|---|
| | | | |
