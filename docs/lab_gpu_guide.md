# Training the 3D CNN on the lab GPU

Step-by-step guide for step 6 ("3D CNN trained"). The laptop has no NVIDIA GPU (decision 5) and
ADNI scans must not go to Colab or other personal cloud services (decision 14), so the CNN is
trained on the university lab GPU. Only the 2 mm cache (about 6 GB) and the tables are copied,
not the 30 GB of processed scans (decision 26).

## 1. Before going (ask the lab admin / supervisor)

- GPU model and memory (6 GB or more is enough for batch size 4).
- Windows or Linux? Can Miniconda be installed in your own user folder?
- Internet access (to install PyTorch) and Git available?
- Can files stay in your own folder until the run is done? How many hours can be booked?
  A full run takes roughly 1 to 3 hours.
- Only team members with approved ADNI access use the folder; data is deleted afterwards.

## 2. On the laptop: build the cache and copy it

```
python -m src.brain_age.cache build --cache-dir D:/fyp-data/cache --workers 4
python -m src.brain_age.cache check --cache-dir D:/fyp-data/cache --workers 4
```

`build` shrinks all 2,489 scans to 2 mm (the exact input the CNN uses) and writes
`manifest.csv` with the size and SHA-256 checksum of every file, including the 6 tables. If
the processed scans or tables ever change, delete the cache folder and build it again.

Copy to a USB drive or external disk (NTFS or exFAT):

- the folder `D:\fyp-data\cache` (contains `2mm\` and `manifest.csv`)
- the folder `data\tables\`

Keep the drive with you; it holds ADNI-derived data.

## 3. At the lab

1. Log in with your university account; open a terminal (Anaconda Prompt on Windows).
2. Check the GPU: `nvidia-smi` (note the GPU name, memory and the CUDA version shown).
3. Get the code (public repo, no login needed):
   ```
   git clone https://github.com/Farrukh776/brain-age-gap-fyp.git
   cd brain-age-gap-fyp
   ```
4. Environment (install Miniconda in your user folder first if needed):
   ```
   conda create -n fyp python=3.11 -y
   conda activate fyp
   ```
   Install the CUDA build of PyTorch with the command pytorch.org gives for the CUDA version
   from `nvidia-smi` (looks like `pip install torch --index-url https://download.pytorch.org/whl/cu12x`),
   then `pip install -r requirements.txt`.
5. Check PyTorch sees the GPU (should print `True` and the GPU name):
   ```
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
6. Copy from the USB: the tables into `data\tables\` inside the repo folder, and the cache
   folder anywhere on the lab disk, e.g. `D:\fyp-cache` (so that `D:\fyp-cache\2mm\` and
   `D:\fyp-cache\manifest.csv` exist).
7. Check the copy is complete and unchanged (must end with "All files match the manifest"):
   ```
   python -m src.brain_age.cache check --cache-dir D:/fyp-cache --workers 4
   ```
8. Smoke test (about a minute; must say `Device cuda`), then delete `outputs\brain_age\cnn\smoke`:
   ```
   python -m src.brain_age.train_cnn --run smoke --smoke --cache-dir D:/fyp-cache
   ```
9. Full run:
   ```
   python -m src.brain_age.train_cnn --run cnn-1 --workers 4 --cache-dir D:/fyp-cache
   ```
   After about 3 epochs press Ctrl+C once, then continue with the same command plus
   `--resume` (tests resuming; also use it if the booking ends mid-run).
10. When it finishes, copy `outputs\brain_age\cnn\cnn-1\` to the USB (history.csv,
    metrics.json, val_predictions.csv, best.pt, last.pt; a few MB).
11. Clean up the lab machine: delete the cache folder, `data\tables\` and `outputs\`.
    Do not commit or push from the lab machine.

## 4. Back on the laptop

Put the `cnn-1` folder in `outputs\brain_age\cnn\`, log the run in `experiment_log.md` and
compare with the ridge baseline (validation MAE 3.79).

## If something goes wrong

- `check` reports missing or changed files: copy those files again (or the whole folder).
- `CUDA out of memory`: close other GPU programs (`nvidia-smi` lists them); if still failing,
  note it and ask before changing the batch size (it is part of decision 25).
- Training says the output folder already has results: use `--resume`, or a new `--run` name.
