"""Train the compact 3D CNN brain-age model on CN training scans; select on CN validation.

Run from the repo root (lab GPU):    python -m src.brain_age.train_cnn --run cnn-1
Continue a stopped run:             python -m src.brain_age.train_cnn --run cnn-1 --resume
Smoke test (laptop CPU, minutes):   python -m src.brain_age.train_cnn --run smoke --smoke
With the 2 mm cache (lab GPU):      add --cache-dir <folder>  (see src/brain_age/cache.py)

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
Prints aggregate numbers only.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
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
    if args.smoke:
        train = train.sample(12, random_state=seed)
        val = val.head(8)
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
    start, best_mae, bad, history = 1, float("inf"), 0, []
    if args.resume:
        state = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        torch.set_rng_state(state["rng"])
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
    json.dump({"run": args.run, "seed": seed, "device": str(device), "smoke": args.smoke,
               "best_epoch": best_epoch, "epochs_run": len(history),
               "target_mean": mean, "target_sd": sd, "cnn_config": ccfg,
               "n_weights": count_weights(model), "val": m},
              open(out / "metrics.json", "w"), indent=2)
    print(f"\nBest epoch {best_epoch}: val n={m['n']} MAE {m['mae']:.2f}  RMSE {m['rmse']:.2f}  "
          f"r {m['pearson_r']:.3f}  mean gap {m['mean_gap']:+.2f}  "
          f"gap-vs-age slope {m['gap_vs_age_slope']:+.2f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
