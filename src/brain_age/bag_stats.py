"""BAG in MCI progressors vs stable subjects, with the test fixed in decision 32 (step 7).

    python -m src.brain_age.bag_stats                 # MCI train + validation (384 subjects)
    python -m src.brain_age.bag_stats --all-splits    # ONLY after E1-E5 are frozen and tested

Reads outputs/brain_age/bag/bag_table.csv (from bag_table.py).
Primary: Welch t-test on the corrected BAG (two-sided), with mean difference and Hedges' g, each
with a bootstrap 95% CI (5,000 resamples, seed 42). Secondary: Mann-Whitney U; the same on the
raw BAG; age per group. Writes (local only) outputs/brain_age/bag/bag_stats_<subset>.json and
bag_by_group_<subset>.png. Prints group summaries only, no subject rows.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import mannwhitneyu, ttest_ind  # noqa: E402

COLORS = {0: "#2a78d6", 1: "#eb6834"}  # categorical slots 1-2 (stable, progressor)
NAMES = {0: "Stable MCI", 1: "Progressive MCI"}
N_BOOT, SEED = 5000, 42


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / sp * (1 - 3 / (4 * (nx + ny) - 9)))


def compare(prog: np.ndarray, stab: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    diffs, gs = [], []
    for _ in range(N_BOOT):
        p = rng.choice(prog, len(prog)), rng.choice(stab, len(stab))
        diffs.append(p[0].mean() - p[1].mean())
        gs.append(hedges_g(*p))
    w = ttest_ind(prog, stab, equal_var=False)
    u = mannwhitneyu(prog, stab, alternative="two-sided")
    return {
        "mean_difference": float(prog.mean() - stab.mean()),
        "mean_difference_ci95": [float(v) for v in np.percentile(diffs, [2.5, 97.5])],
        "hedges_g": hedges_g(prog, stab),
        "hedges_g_ci95": [float(v) for v in np.percentile(gs, [2.5, 97.5])],
        "welch_t": float(w.statistic), "welch_df": float(w.df), "welch_p": float(w.pvalue),
        "mann_whitney_u": float(u.statistic), "mann_whitney_p": float(u.pvalue),
        # probability a random progressor has a higher BAG than a random stable subject
        "common_language_effect": float(u.statistic / (len(prog) * len(stab))),
    }


def describe(x: np.ndarray) -> dict:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "median": float(med), "iqr": [float(q1), float(q3)]}


def plot(t: pd.DataFrame, res: dict, path: Path, subset: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    rng = np.random.default_rng(SEED)
    for pos, lab in enumerate([0, 1]):
        v = t.loc[t["label"] == lab, "bag"].to_numpy()
        ax.boxplot(v, positions=[pos], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor="#555", linewidth=1.2),
                   medianprops=dict(color="#222", linewidth=2),
                   whiskerprops=dict(color="#777"), capprops=dict(color="#777"))
        ax.scatter(pos + rng.uniform(-0.17, 0.17, len(v)), v, s=14, alpha=0.55,
                   color=COLORS[lab], edgecolors="white", linewidths=0.6, zorder=3)
        ax.text(pos, v.max() + 1.2,
                f"mean {v.mean():+.1f}\nn = {len(v)}", ha="center", va="bottom",
                fontsize=9, color="#333")
    ax.axhline(0, color="#999", ls="--", lw=1, zorder=1)
    ax.set_xticks([0, 1], [NAMES[0], NAMES[1]])
    ax.set_ylabel("Brain age gap, bias-corrected (years)")
    p = res["corrected"]
    ax.set_title(f"BAG by 36-month outcome, MCI {subset}\n"
                 f"difference {p['mean_difference']:+.2f} y "
                 f"[{p['mean_difference_ci95'][0]:.2f}, {p['mean_difference_ci95'][1]:.2f}], "
                 f"Hedges' g {p['hedges_g']:.2f}, Welch p = {p['welch_p']:.1g}", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/brain_age.yaml")
    parser.add_argument("--all-splits", action="store_true",
                        help="include MCI test subjects: only after E1-E5 are frozen (decision 32)")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dest = Path(cfg["paths"]["out_dir"]) / "bag"
    t = pd.read_csv(dest / "bag_table.csv")
    subset = "all splits" if args.all_splits else "train + validation"
    if not args.all_splits:
        t = t[t["split"].isin(["train", "val"])]
    tag = subset.replace(" + ", "_").replace(" ", "_")

    prog, stab = t[t["label"] == 1], t[t["label"] == 0]
    res = {"subset": subset, "n_subjects": len(t), "decision": 32,
           "groups": {NAMES[k]: {"bag": describe(g["bag"].to_numpy()),
                                 "bag_raw": describe(g["bag_raw"].to_numpy()),
                                 "age": describe(g["age"].to_numpy())}
                      for k, g in [(0, stab), (1, prog)]},
           "corrected": compare(prog["bag"].to_numpy(), stab["bag"].to_numpy()),
           "raw": compare(prog["bag_raw"].to_numpy(), stab["bag_raw"].to_numpy()),
           "age": compare(prog["age"].to_numpy(), stab["age"].to_numpy())}
    json.dump(res, open(dest / f"bag_stats_{tag}.json", "w"), indent=2)
    plot(t, res, dest / f"bag_by_group_{tag}.png", subset)

    print(f"MCI {subset}: {len(t)} subjects")
    for name, g in res["groups"].items():
        b, a = g["bag"], g["age"]
        print(f"  {name:16s} n={b['n']:3d}  BAG mean {b['mean']:+.2f} (SD {b['sd']:.2f}), "
              f"median {b['median']:+.2f}  | raw BAG mean {g['bag_raw']['mean']:+.2f}  "
              f"| age {a['mean']:.1f} (SD {a['sd']:.1f})")
    for key, label in [("corrected", "Corrected BAG (primary)"), ("raw", "Raw BAG"), ("age", "Age")]:
        r = res[key]
        print(f"{label}: difference {r['mean_difference']:+.2f} "
              f"[{r['mean_difference_ci95'][0]:+.2f}, {r['mean_difference_ci95'][1]:+.2f}], "
              f"Hedges' g {r['hedges_g']:.2f} [{r['hedges_g_ci95'][0]:.2f}, "
              f"{r['hedges_g_ci95'][1]:.2f}], Welch t({r['welch_df']:.0f}) = {r['welch_t']:.2f}, "
              f"p = {r['welch_p']:.2g}; Mann-Whitney p = {r['mann_whitney_p']:.2g}; "
              f"P(progressor > stable) = {r['common_language_effect']:.2f}")
    print(f"Saved {dest / f'bag_stats_{tag}.json'} and bag_by_group_{tag}.png")


if __name__ == "__main__":
    main()
