"""Train the compact 3D CNN brain-age model on CN training scans; select on CN validation.

Run from the repo root (lab GPU):    python -m src.brain_age.train_cnn --run cnn-1
Continue a stopped run:             python -m src.brain_age.train_cnn --run cnn-1 --resume
Smoke test (laptop CPU, minutes):   python -m src.brain_age.train_cnn --run smoke --smoke
With the 2 mm cache (lab GPU):      add --cache-dir <folder>  (see src/brain_age/cache.py)
One cross-validation fold (0-4):    python -m src.brain_age.train_cnn --run cnn-cv --fold 0

--fold k (decision 30): the training scans are split into the same 5 subject-grouped folds as
the ridge baseline (GroupKFold, shuffle, seed 42); the model trains on the other 4 folds,
early-stops on CN validation like cnn-1, and predicts fold k, which it never saw. The five
fold runs give out-of-fold training predictions (for the age-bias correction, fitted on
training data only) and five models to average. Output goes to cnn/<run>/fold<k>/.

Data: src/brain_age/data.py (all CN training visits with flip + shift augmentation;
CN validation baselines without augmentation). The test set is not loaded.
Target: age scaled with the training-scan mean and SD; loss L1 (mean absolute error).
Each epoch: train, then validation MAE in years. The best epoch is kept; training stops when
validation MAE has not improved for `early_stopping_patience` epochs. Learning rate is halved
after `lr_patience` epochs without improvement.

Writes (local only, outputs/brain_age/cnn/<run>/):
  history.csv          per epoch: train loss, train MAE, val MAE, learning rate, seconds
  best.pt / last.pt    model weights (best epoch) / full state for --resume
  val_predictions.csv  validation predictions of the best epoch
  metrics.json         settings and validation metrics of the best epoch
  oof_predictions.csv  (--fold only) predictions for the held-out training fold
Prints aggregate numbers only.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold
from torch import nn

from src.brain_age.cnn import BrainAgeCNN, count_weights
from src.brain_age.data import BrainAgeDataset, make_loader, scan_table
from src.brain_age.train_baseline import metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, loader, device, mean, sd) -> tuple:
    model.eval()
    preds, idx = [], []
    with torch.no_grad():
        for images, _, i in loader:
            preds.append(model(images.to(device)).cpu() * sd + mean)
            idx.append(i)
    order = torch.cat(idx).argsort()
    return torch.cat(preds)[order].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/brain_age.yaml")
    parser.add_argument("--run", required=True, help="run name, e.g. cnn-1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="12 training and 8 validation scans, 2 epochs (tests the code)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-dir", help="2 mm cache from src.brain_age.cache "
                        "(overrides cnn.cache_dir; on the lab GPU run 'cache check' first)")
    parser.add_argument("--fold", type=int, help="0 to 4: train on the other training folds and "
                        "predict this one (out-of-fold predictions, decision 30)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ccfg = cfg["cnn"]
    prep = yaml.safe_load(open(cfg["paths"]["preprocessing_config"]))
    seed = cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    table = scan_table(cfg)
    cn = table[table["cohort"] == "CN"]
    train = cn[cn["split"] == "train"]
    val = cn[cn["split"] == "val"]
    epochs = ccfg["epochs"]
    held = None
    if args.fold is not None:
        n_folds = cfg["baseline"]["cv_folds"]
        if not 0 <= args.fold < n_folds:
            raise SystemExit(f"--fold must be 0 to {n_folds - 1}")
        cv = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        tr_idx, te_idx = list(cv.split(train, groups=train["RID"]))[args.fold]
        train, held = train.iloc[tr_idx], train.iloc[te_idx]
        if set(train["RID"]) & set(held["RID"]):
            raise SystemExit("a subject is in both the training folds and the held-out fold")
        # Short fingerprint of the held-out scans: the same on every machine if folds match
        tag = hashlib.sha256(",".join(sorted(held["image_id"].astype(str))).encode()).hexdigest()
        print(f"Fold {args.fold} of {n_folds}: train {len(train)} scans, held out {len(held)} "
              f"scans ({held['RID'].nunique()} subjects), fingerprint {tag[:12]}")
    if args.smoke:
        train = train.sample(12, random_state=seed)
        val = val.head(8)
        held = held.head(8) if held is not None else None
        epochs = 2
    if set(train["RID"]) & set(val["RID"]):
        raise SystemExit("a subject is in both train and validation")

    mean, sd = float(train["age"].mean()), float(train["age"].std())
    factor = int(ccfg["input_downsample_mm"])
    cache = args.cache_dir or ccfg.get("cache_dir")
    if cache:
        ccfg["cache_dir"] = cache  # recorded in metrics.json
    ds_train = BrainAgeDataset(train, prep["paths"]["out_dir"], factor, augment=True,
                               max_shift=ccfg["augment_max_shift"], cache_dir=cache)
    ds_val = BrainAgeDataset(val, prep["paths"]["out_dir"], factor, augment=False,
                             cache_dir=cache)
    bs = ccfg["batch_size"]

    model = BrainAgeCNN(dropout=ccfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ccfg["lr"],
                            weight_decay=ccfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=ccfg["lr_patience"])
    loss_fn = nn.L1Loss()

    out = Path(cfg["paths"]["out_dir"]) / "cnn" / args.run
    if args.fold is not None:
        out = out / f"fold{args.fold}"
    start, best_mae, bad, history = 1, float("inf"), 0, []
    if args.resume:
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        torch.set_rng_state(state["rng"].cpu())  # map_location moved it to the GPU
        if "cuda_rng" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng"]])
        start, best_mae, bad = state["epoch"] + 1, state["best_mae"], state["bad"]
        history = pd.read_csv(out / "history.csv").to_dict("records")
        print(f"Resuming {args.run} at epoch {start} (best val MAE {best_mae:.2f})")
    elif out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already has results; use a new --run name or --resume")
    out.mkdir(parents=True, exist_ok=True)
    # Shuffle order depends on the start epoch, so a resumed run does not repeat epoch 1's order
    train_loader = make_loader(ds_train, bs, True, args.workers, seed + start - 1)
    val_loader = make_loader(ds_val, bs, False, args.workers, seed)

    print(f"Device {device}; {count_weights(model):,} weights; train {len(train)} scans "
          f"({train['RID'].nunique()} subjects), val {len(val)} scans; "
          f"target mean {mean:.1f}, SD {sd:.1f} years")

    for epoch in range(start, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, abs_sum, n = 0.0, 0.0, 0
        for images, ages, _ in train_loader:
            images, ages = images.to(device), ages.to(device)
            pred = model(images)
            loss = loss_fn(pred, (ages - mean) / sd)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(ages)
            abs_sum += ((pred.detach() * sd + mean) - ages).abs().sum().item()
            n += len(ages)

        pred_val = predict(model, val_loader, device, mean, sd)
        val_mae = float(np.abs(pred_val - val["age"].to_numpy()).mean())
        sched.step(val_mae)
        improved = val_mae < best_mae
        if improved:
            best_mae, bad = val_mae, 0
            torch.save(model.state_dict(), out / "best.pt")
            best_pred = pred_val
        else:
            bad += 1
        lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": loss_sum / n, "train_mae": abs_sum / n,
                        "val_mae": val_mae, "lr": lr, "seconds": time.time() - t0})
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "epoch": epoch, "best_mae": best_mae, "bad": bad}, out / "last.pt")
        print(f"epoch {epoch:3d}  train MAE {abs_sum / n:5.2f}  val MAE {val_mae:5.2f}"
              f"{' *' if improved else '  '}  lr {lr:.1e}  {time.time() - t0:5.0f} s")
        if bad >= ccfg["early_stopping_patience"]:
            print(f"Stopping: no improvement for {bad} epochs")
            break

    # Final metrics from the best weights (also correct after --resume)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    best_pred = predict(model, val_loader, device, mean, sd)
    m = metrics(val["age"].to_numpy(), best_pred)
    best_epoch = int(pd.DataFrame(history)["val_mae"].idxmin()) + 1
    val[["image_id", "RID", "age"]].assign(pred_cnn=best_pred).to_csv(
        out / "val_predictions.csv", index=False)
    result = {"run": args.run, "fold": args.fold, "seed": seed, "device": str(device),
              "smoke": args.smoke, "best_epoch": best_epoch, "epochs_run": len(history),
              "target_mean": mean, "target_sd": sd, "cnn_config": ccfg,
              "n_weights": count_weights(model), "val": m}
    if held is not None:
        ds_held = BrainAgeDataset(held, prep["paths"]["out_dir"], factor, augment=False,
                                  cache_dir=cache)
        pred_held = predict(model, make_loader(ds_held, bs, False, args.workers, seed),
                            device, mean, sd)
        held[["image_id", "RID", "age"]].assign(pred_cnn=pred_held, fold=args.fold).to_csv(
            out / "oof_predictions.csv", index=False)
        result["held_out"] = mh = metrics(held["age"].to_numpy(), pred_held)
        print(f"Held-out fold {args.fold}: n={mh['n']} MAE {mh['mae']:.2f}  "
              f"r {mh['pearson_r']:.3f}  gap-vs-age slope {mh['gap_vs_age_slope']:+.2f}")
    json.dump(result, open(out / "metrics.json", "w"), indent=2)
    print(f"\nBest epoch {best_epoch}: val n={m['n']} MAE {m['mae']:.2f}  RMSE {m['rmse']:.2f}  "
          f"r {m['pearson_r']:.3f}  mean gap {m['mean_gap']:+.2f}  "
          f"gap-vs-age slope {m['gap_vs_age_slope']:+.2f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
