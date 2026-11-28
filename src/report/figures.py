"""Figures for the FYP-I report (step 10). No model is run; everything comes from saved results.

Run from the repo root:  python -m src.report.figures

Writes PNG files (300 dpi) to outputs/figures/ (local only). Figures that show individual
subjects as points (brain-age scatter, BAG by group) stay there; summary figures (cohort flow,
ROC curves) are also copied to docs/figures/ for the repository. No subject IDs appear anywhere.

  cohort_flow.png       subjects from ADNI to the final cohorts (counts: cohort_definition.md)
  brain_age_test.png    frozen brain-age model on the CN test set: predicted vs actual age,
                        and the brain-age gap against age before and after bias correction
  bag_groups.png        corrected BAG, stable vs progressor (MCI train + validation, decision 32)
  roc_curves.png        E1 vs E2 ROC curves (logistic regression) on validation and on test
"""
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

OUT = Path("outputs/figures")
DOCS = Path("docs/figures")

# Reference palette (dataviz skill), light mode: categorical slots 1-2, text and grid tokens
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name: str, public: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    if public:
        DOCS.mkdir(parents=True, exist_ok=True)
        shutil.copy(OUT / name, DOCS / name)
    print(f"  {name}{'  (also docs/figures)' if public else ''}")


# ---------------------------------------------------------------- cohort flow

def box(ax, x, y, w, h, text, bold=False, fill="#f4f3f0", size=8.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.06",
                                fc=fill, ec=MUTED, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", color=INK, fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def side(ax, x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.06",
                                fc=SURFACE, ec=GRID, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", color=INK2, fontsize=7.5, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0))


def cohort_flow() -> None:
    # drawn at print size: 1 axis unit = 1 inch, 7.2 in wide (inserted at about 6.3 in)
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    ax.grid(False)
    nl = chr(10)
    w = 2.5
    cols = {"CN": 1.6, "MCI": 5.0}
    rows = [(5.4, 0.62), (3.95, 0.5), (2.6, 0.5)]          # (y, height) of the main boxes
    steps = {
        "CN": [f"Cognitively unimpaired (CN){nl}at baseline: 1,593", "947 subjects", "935 subjects"],
        "MCI": [f"Mild cognitive impairment (MCI){nl}at baseline: 1,550", "454 subjects", "453 subjects"],
    }
    excl = {
        "CN": [nl.join(["Later MCI or AD: 215", "No 3 T scan in 90 days: 431"]),
               nl.join(["Failed scan QC: 8", "Enrolled twice: 4"])],
        "MCI": [nl.join(["Enrolled in ADNI4: 234", "Reverted to CN: 114",
                         "Follow-up < 36 months: 486", "No 3 T scan in 90 days: 262"]),
                "Failed scan QC: 1"],
    }
    ew = 1.75
    for c, x in cols.items():
        for i, ((y, h), text) in enumerate(zip(rows, steps[c])):
            box(ax, x, y, w, h, text, bold=(i == 0), size=8)
            if i:
                py, ph = rows[i - 1]
                arrow(ax, x, py - ph / 2, x, y + h / 2)
                ey = ((py - ph / 2) + (y + h / 2)) / 2
                lines = excl[c][i - 1].count(nl) + 1
                ex = x + 0.2 + ew / 2
                side(ax, ex, ey, ew, 0.12 + 0.155 * lines, excl[c][i - 1])
                ax.plot([x, ex - ew / 2], [ey, ey], color=MUTED, lw=0.8)
    box(ax, 1.6, 1.25, 2.95, 0.85, nl.join(["Brain-age cohort (CN)", "Train 654 subjects (1,755 scans)",
                                           "Validation 140  ·  Test 141"]), fill="#eaf1fb", size=7.8)
    box(ax, 5.0, 1.25, 2.95, 0.85, nl.join(["Progression cohort (MCI)", "Train 316  ·  Validation 68  ·  Test 69",
                                           "299 stable  ·  154 progressors (34%)"]), fill="#fdeee7", size=7.8)
    for x in cols.values():
        arrow(ax, x, rows[2][0] - rows[2][1] / 2, x, 1.25 + 0.425)
    ax.text(3.6, 0.32, nl.join([
        "Subject-level 70 / 15 / 15 splits (seed 42); no person in more than one split or cohort.",
        "3 T scans only; ADNI1, GO, 2 and 3 (CN also ADNI4). Progressor = dementia within 36 months."]),
        ha="center", va="center", color=INK2, fontsize=7.3, linespacing=1.4)
    save(fig, "cohort_flow.png", public=True)


# ---------------------------------------------------------------- brain age

def brain_age_test() -> None:
    t = pd.read_csv("outputs/brain_age/final/cn_test_predictions.csv")
    age, pred = t["age"].to_numpy(), t["brain_age"].to_numpy()
    mae = np.abs(pred - age).mean()
    r = np.corrcoef(age, pred)[0, 1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3))

    lo, hi = np.floor(min(age.min(), pred.min()) - 2), np.ceil(max(age.max(), pred.max()) + 2)
    a.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    a.scatter(age, pred, s=22, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=0.6, zorder=2)
    a.set_xlim(lo, hi)
    a.set_ylim(lo, hi)
    a.set_aspect("equal")
    a.set_xlabel("Chronological age (years)")
    a.set_ylabel("Predicted brain age (years)")
    a.set_title("a  Predicted vs actual age", loc="left", color=INK, fontsize=9.5)
    a.text(0.04, 0.95, f"CN test set, n = {len(age)}\nMAE {mae:.2f} years, r = {r:.2f}",
           transform=a.transAxes, va="top", color=INK2, fontsize=8)
    a.text(hi - 0.5, hi - 1.5, "identity", color=MUTED, fontsize=7, ha="right", rotation=45, rotation_mode="anchor")

    xs = np.linspace(age.min(), age.max(), 50)
    for col, color, marker, name in (("bag_raw", ORANGE, "^", "Before correction"),
                                     ("bag", BLUE, "o", "After correction")):
        g = t[col].to_numpy()
        slope, icpt = np.polyfit(age, g, 1)
        b.scatter(age, g, s=18, color=color, marker=marker, alpha=0.6, edgecolor=SURFACE, linewidth=0.5)
        b.plot(xs, slope * xs + icpt, color=color, lw=2, label=f"{name} (slope {slope:+.2f})")
    b.axhline(0, color=MUTED, lw=1)
    b.set_xlabel("Chronological age (years)")
    b.set_ylabel("Brain-age gap (years)")
    b.set_title("b  Age bias before and after correction", loc="left", color=INK, fontsize=9.5)
    b.legend(loc="upper right", fontsize=7.6, labelcolor=INK2)
    fig.tight_layout(w_pad=2.5)
    save(fig, "brain_age_test.png", public=False)


# ---------------------------------------------------------------- BAG by group

def bag_groups() -> None:
    t = pd.read_csv("outputs/brain_age/bag/bag_table.csv")
    t = t[t["split"].isin(["train", "val"])]          # decision 32: MCI test waits for E1-E5
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    groups = [(0, "Stable", BLUE, "o"), (1, "Progressor", ORANGE, "^")]
    for pos, (label, name, color, marker) in enumerate(groups):
        g = t.loc[t["label"] == label, "bag"].to_numpy()
        ax.boxplot(g, positions=[pos], widths=0.42, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=SURFACE, edgecolor=INK2, lw=0.9),
                   medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=INK2, lw=0.9),
                   capprops=dict(color=INK2, lw=0.9))
        ax.scatter(pos + rng.uniform(-0.17, 0.17, len(g)), g, s=12, color=color, marker=marker,
                   alpha=0.5, edgecolor="none", zorder=3)
        ax.text(pos, ax.get_ylim()[1] if False else 0, "", color=INK2)
        ax.annotate(f"mean {g.mean():+.2f}\nn = {len(g)}", xy=(pos + 0.27, np.median(g)),
                    va="center", color=INK2, fontsize=7.6)
    ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.set_xticks([0, 1], [g[1] for g in groups])
    ax.set_xlim(-0.6, 1.85)
    ax.set_ylabel("Corrected brain-age gap (years)")
    ax.set_title("Brain-age gap by 36-month outcome\n(MCI train + validation)", loc="left",
                 color=INK, fontsize=9.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "bag_groups.png", public=False)


# ---------------------------------------------------------------- ROC

def roc_curves() -> None:
    val1 = pd.read_csv("outputs/progression/E1/val_predictions.csv")
    val2 = pd.read_csv("outputs/progression/E2/val_predictions.csv")
    test = pd.read_csv("outputs/progression/test/test_predictions.csv")
    sets = [("a  Validation (n = 68)", val1["label"], val1["prob_logistic_regression"],
             val2["prob_logistic_regression"]),
            ("b  Test, used once (n = 69)", test["label"], test["E1_logistic_regression"],
             test["E2_logistic_regression"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, (title, y, p1, p2) in zip(axes, sets):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)))
        for p, name, color, ls in ((p1, "E1  MRI", BLUE, "-"), (p2, "E2  MRI + age", ORANGE, (0, (5, 2)))):
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=color, lw=2, ls=ls, label=f"{name}  (AUC {roc_auc_score(y, p):.3f})",
                    drawstyle="steps-post")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        ax.legend(loc="lower right", fontsize=7.6, labelcolor=INK2)
        ax.text(0.62, 0.55, "chance", color=MUTED, fontsize=7, rotation=45)
    fig.suptitle("MCI-to-AD progression, logistic regression (frozen models)", x=0.06, ha="left",
                 color=INK, fontsize=10)
    fig.tight_layout(w_pad=3)
    save(fig, "roc_curves.png", public=True)


def main() -> None:
    print("Writing report figures:")
    cohort_flow()
    brain_age_test()
    bag_groups()
    roc_curves()


if __name__ == "__main__":
    main()
