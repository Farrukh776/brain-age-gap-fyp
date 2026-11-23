# Training the 3D CNN on a GPU (Colab or lab)

Step-by-step guide for step 6 ("3D CNN trained"). The laptop has no NVIDIA GPU (decision 5).
Only the 2 mm cache (about 6 GB) and the tables are moved, not the 30 GB of processed scans
(decision 26), as one zip through the university Google Drive (decision 27). The supervisor
approved Google Colab with the university account (decision 28); the lab GPU is the backup.

Rules (ADNI Data Use Agreement): use only the **university** Google account (or the one personal
account approved in decision 29), never any other personal
one; the Drive folder is shared only with team members who have ADNI access; never
"anyone with the link"; delete the data when the CNN work is finished.

## 1. On the laptop: build the cache, pack it, upload it

```
python -m src.brain_age.cache build --cache-dir D:/fyp-data/cache --workers 4
python -m src.brain_age.cache check --cache-dir D:/fyp-data/cache --workers 4
python -m src.brain_age.cache pack  --cache-dir D:/fyp-data/cache --zip D:/fyp-data/upload/fyp_cache_2mm.zip
```

`build` shrinks all 2,489 scans to 2 mm (the exact input the CNN uses) and writes
`manifest.csv` with the size and SHA-256 checksum of every file, including the 6 tables. `pack`
puts the cache and tables in one zip. If the processed scans or tables ever change, delete the
cache folder and the zip and build both again.

Upload:

1. Sign in to drive.google.com with the university account.
2. In My Drive, make a folder named exactly `fyp-adni-private`. Share it only with team
   members who have ADNI access (by their university emails).
3. Drag `fyp_cache_2mm.zip` into that folder and wait until the upload box (bottom right) says
   it is complete. Keep the laptop awake meanwhile.

## 2. Train on Colab (main route)

1. Go to colab.research.google.com, signed in with the **university** account. New notebook.
2. Runtime > Change runtime type > **T4 GPU** > Save.
3. Run these cells one by one (a cell runs with Shift+Enter). Drive asks for permission once:
   choose the university account.

   ```
   # Cell 1: connect Drive, get the code, install the MRI packages
   from google.colab import drive
   drive.mount('/content/drive')
   !git clone https://github.com/Farrukh776/brain-age-gap-fyp.git /content/brain-age-gap-fyp
   %cd /content/brain-age-gap-fyp
   !pip install -q nibabel nilearn SimpleITK
   ```

   ```
   # Cell 2: copy the zip to the Colab disk, unpack and check (must end "All files match")
   !cp /content/drive/MyDrive/fyp-adni-private/fyp_cache_2mm.zip /content/
   !python -m src.brain_age.cache unpack --zip /content/fyp_cache_2mm.zip --cache-dir /content/cache --workers 2
   !rm /content/fyp_cache_2mm.zip
   ```

   ```
   # Cell 3: save CNN results straight into the private Drive folder (survives disconnects)
   !mkdir -p /content/drive/MyDrive/fyp-adni-private/cnn outputs/brain_age
   !ln -sfn /content/drive/MyDrive/fyp-adni-private/cnn outputs/brain_age/cnn
   import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
   ```

   ```
   # Cell 4: smoke test (about a minute; must say "Device cuda"), then remove it
   !python -m src.brain_age.train_cnn --run smoke --smoke --workers 2 --cache-dir /content/cache
   !rm -r outputs/brain_age/cnn/smoke
   ```

   ```
   # Cell 5: full run (roughly 15 to 60 minutes)
   !python -m src.brain_age.train_cnn --run cnn-1 --workers 2 --cache-dir /content/cache
   ```

   Test resuming once: after about 3 epochs press the stop button of cell 5, then run it again
   with `--resume` added at the end. It must continue from the next epoch.
4. **If Colab disconnects** (free sessions can end without warning): open the notebook again,
   run cells 1 to 3, then cell 5 with `--resume`. Each epoch is saved to Drive, so at most one
   epoch is lost.
5. When it finishes, the results are in Drive at `fyp-adni-private/cnn/cnn-1/` (history.csv,
   metrics.json, val_predictions.csv, best.pt, last.pt). Download that folder to the laptop.
   Runtime > Disconnect and delete runtime (removes the unpacked scans from Colab).

If a teammate runs it from their account: the shared folder is under "Shared with me";
right-click it > Organize > Add shortcut > My Drive, so the path above works.

**Farrukh's personal Gmail (decision 29, supervisor-approved for this account only).** Use it
when the university account gets "failed to assign a backend".
1. University account, drive.google.com: right-click `fyp-adni-private` > Share > add the
   personal Gmail as **Editor** > Send. (If Drive says sharing outside the university is not
   allowed: download the zip and upload it into a private `fyp-adni-private` folder in the
   personal Drive instead, then skip step 2.)
2. Personal account, drive.google.com > Shared with me > right-click `fyp-adni-private` >
   Organize > Add shortcut > My Drive.
3. Personal account in Colab (check the account icon top right), T4 GPU, then cells 1 to 5
   unchanged; in cell 1 allow Drive access for the personal account.
4. Clean-up when the CNN work is final: delete the results and any zip copy from the personal
   Drive, empty its trash, remove the share from the university folder, and delete the Colab
   notebook's outputs (Edit > Clear all outputs).

## 3. Back on the laptop

Put the `cnn-1` folder in `outputs\brain_age\cnn\`, log the run in `experiment_log.md` and
compare with the ridge baseline (validation MAE 3.79). When all CNN work is finished, delete
the zip (and any results you no longer need) from Drive.

## Backup: lab GPU

Ask first: GPU model and memory (6 GB or more), Windows or Linux, can Miniconda be installed
in your user folder, internet access, can files stay until the run is done, booking hours.

1. Log in with the university account; open a terminal (Anaconda Prompt on Windows).
2. `nvidia-smi` (note GPU name, memory and CUDA version).
3. `git clone https://github.com/Farrukh776/brain-age-gap-fyp.git` then `cd brain-age-gap-fyp`.
4. `conda create -n fyp python=3.11 -y`, `conda activate fyp`, install the CUDA build of PyTorch
   with the command pytorch.org gives for that CUDA version, then `pip install -r requirements.txt`.
5. `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
6. Download the zip from the university Drive into your folder, e.g. `D:\fyp-download\`, then:
   ```
   python -m src.brain_age.cache unpack --zip D:/fyp-download/fyp_cache_2mm.zip --cache-dir D:/fyp-cache --workers 4
   python -m src.brain_age.train_cnn --run smoke --smoke --cache-dir D:/fyp-cache
   python -m src.brain_age.train_cnn --run cnn-1 --workers 4 --cache-dir D:/fyp-cache
   ```
   (delete `outputs\brain_age\cnn\smoke` after the smoke test; use `--resume` if stopped).
7. Upload `outputs\brain_age\cnn\cnn-1\` to the private Drive folder, then delete the zip, the
   cache folder, `data\tables\` and `outputs\` from the lab PC. Do not commit or push there.

## If something goes wrong

- Cell 1 prints "ERROR: pip's dependency resolver ... google-colab requires pandas==2.2.3, but
  you have pandas 3.x": expected and harmless. nilearn pulls in pandas 3, the same version the
  code was tested with on the laptop; only Colab's own extras want the older one. Continue.

- `check` reports missing or changed files: download / copy the zip again and unpack again.
- `CUDA out of memory`: close other GPU programs; if still failing, note it and ask before
  changing the batch size (it is part of decision 25).
- Training says the output folder already has results: use `--resume`, or a new `--run` name.
- Colab says no GPU is available (free quota used up): try again later, or use the lab.
