"""Design diagrams for the FYP-I report (step 10). They contain no data, only structure.

Run from the repo root:  python -m src.report.diagrams

Every diagram is drawn at its printed size (7.2 in wide, inserted at about 6.3 in) with one
axis unit = one inch, so text stays legible in print (about 7-8 pt on paper).
Writes PNG files (300 dpi) to docs/figures/:
  pipeline_overview.png   solution overview: FYP-I steps (done) and FYP-II steps (planned)
  architecture.png        code modules, configs, data stores and compute
  data_design.png         the main tables and how they link (stands in for an ER diagram)
  freeze_workflow.png     train -> validate -> freeze -> test once
  gantt.png               timeline: FYP-I (done) and FYP-II (months, dates to be confirmed)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path("docs/figures")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
DONE_FILL, DATA_FILL, NEUTRAL = "#eaf1fb", "#fdeee7", "#f4f3f0"
NL = chr(10)
FS = 8.5          # main text size (pt) at 7.2 in width

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300, "savefig.bbox": "tight",
                     "figure.facecolor": SURFACE})


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fill=NEUTRAL, size=FS, bold=False, dashed=False, family=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.0,rounding_size=0.06", fc=fill, ec=MUTED,
                                lw=0.9, ls=(0, (4, 2)) if dashed else "-"))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.3,
            fontweight="bold" if bold else "normal", family=family)


def arrow(ax, x0, y0, x1, y1, both=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=MUTED, lw=1.0,
                                shrinkA=0, shrinkB=0))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------- 1. pipeline overview

def pipeline_overview():
    fig, ax = canvas(7.2, 4.5)
    w, h = 2.15, 0.9
    xs = [1.2, 3.6, 6.0]
    ys = [3.85, 2.5, 1.15]
    done = [("ADNI data" + NL + "T1 MRI + clinical tables"),
            ("Cohorts and splits" + NL + "CN: brain age; MCI: outcome" + NL + "subject-level 70/15/15"),
            ("MRI preprocessing + QC" + NL + "N4, affine to MNI," + NL + "every scan checked by eye"),
            ("Brain-age model (CN only)" + NL + "ridge baseline, 3D CNN," + NL + "age-bias correction"),
            ("Brain age gap (BAG)" + NL + "for every MCI subject" + NL + "BAG = brain age - age"),
            ("E1: MRI features" + NL + "E2: MRI + age" + NL + "frozen, tested once")]
    plan = [("E3: MRI + BAG" + NL + "E4: MRI + age + BAG" + NL + "E5: + clinical (optional)"),
            ("Trust layer" + NL + "explanations (XAI)" + NL + "and calibration")]
    pos = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[2], ys[0]),     # row 1, left to right
           (xs[2], ys[1]), (xs[1], ys[1]), (xs[0], ys[1]),     # row 2, right to left
           (xs[0], ys[2]), (xs[1], ys[2])]                     # row 3, left to right
    texts = done + plan
    for i, ((x, y), t) in enumerate(zip(pos, texts)):
        box(ax, x, y, w, h, t, fill=DONE_FILL if i < 6 else SURFACE, dashed=i >= 6)
        if i:
            (px, py) = pos[i - 1]
            if py == y:
                d = 1 if x > px else -1
                arrow(ax, px + d * w / 2, y, x - d * w / 2, y)
            else:
                arrow(ax, x, py - h / 2, x, y + h / 2)
    # legend in the free slot
    lx, ly = xs[2] - 0.9, ys[2] + 0.2
    ax.add_patch(FancyBboxPatch((lx, ly - 0.09), 0.3, 0.18, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=DONE_FILL, ec=MUTED, lw=0.9))
    ax.text(lx + 0.4, ly, "FYP-I (done)", va="center", fontsize=FS - 0.5, color=INK2)
    ax.add_patch(FancyBboxPatch((lx, ly - 0.49), 0.3, 0.18, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=SURFACE, ec=MUTED, lw=0.9, ls=(0, (4, 2))))
    ax.text(lx + 0.4, ly - 0.4, "FYP-II (planned)", va="center", fontsize=FS - 0.5, color=INK2)
    ax.text(3.6, 0.25, "Primary question: does BAG add predictive value beyond chronological age? (E4 vs E2)",
            ha="center", fontsize=FS - 0.5, color=INK2, style="italic")
    save(fig, "pipeline_overview.png")


# ---------------------------------------------------------------- 2. architecture

def architecture():
    fig, ax = canvas(7.2, 7.3)
    ax.text(3.6, 7.05, "Python code in the git repository (public: code and documents only)",
            ha="center", fontsize=FS, fontweight="bold", color=INK)
    mods = [("src/cohort", "cohorts, labels, splits"),
            ("src/preprocessing", "N4, affine to MNI, QC sheets, checks"),
            ("src/brain_age", "features, ridge, 3D CNN, correction, BAG"),
            ("src/features", "MCI feature table for E1-E5"),
            ("src/progression", "E1/E2 training, frozen evaluation"),
            ("src/report", "report figures and diagrams")]
    mx, mw, mh = 2.0, 3.6, 0.56
    for i, (name, desc) in enumerate(mods):
        box(ax, mx, 6.45 - i * 0.68, mw, mh, name + NL + desc, fill=DONE_FILL)
    data = [("ADNI data (never in git)" + NL + "raw scans, clinical tables," + NL + "processed scans", 6.1),
            ("data/tables, data/features" + NL + "cohorts, splits, QC tables," + NL + "voxel features", 4.75),
            ("outputs/ (local only)" + NL + "models, predictions," + NL + "metrics, figures", 3.4)]
    for t, y in data:
        box(ax, 6.0, y, 2.25, 1.0, t, fill=DATA_FILL)
        arrow(ax, mx + mw / 2, y, 6.0 - 2.25 / 2, y, both=True)
    support = [("configs/*.yaml" + NL + "paths, grids, seed 42," + NL + "frozen model hashes", 1.2),
               ("tests/ (pytest)" + NL + "split overlap, E1 = E2" + NL + "except age, test lock", 3.6),
               ("docs/" + NL + "decision, experiment" + NL + "and session logs", 6.0)]
    for t, x in support:
        box(ax, x, 1.95, 2.2, 0.9, t)
    ax.text(0.15, 1.07, "All code runs on:", fontsize=FS - 0.5, color=INK2, va="center")
    box(ax, 2.3, 0.5, 3.1, 0.6, "Laptop (CPU)" + NL + "every step except CNN training")
    box(ax, 5.65, 0.5, 2.9, 0.6, "Google Colab GPU" + NL + "3D CNN training (decisions 26-29)")
    save(fig, "architecture.png")


# ---------------------------------------------------------------- 3. data design

def table(ax, x, y_top, title, fields, fill, w=2.2):
    row, head = 0.22, 0.3
    h = head + row * len(fields) + 0.08
    ax.add_patch(FancyBboxPatch((x - w / 2, y_top - h), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=SURFACE, ec=MUTED, lw=0.9))
    ax.add_patch(FancyBboxPatch((x - w / 2, y_top - head), w, head, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fill, ec=MUTED, lw=0.9))
    ax.text(x, y_top - head / 2, title, ha="center", va="center", fontsize=FS - 0.5, fontweight="bold", color=INK)
    for i, f in enumerate(fields):
        ax.text(x - w / 2 + 0.1, y_top - head - 0.13 - row * i, f, ha="left", va="center",
                fontsize=FS - 1.5, color=INK2, family="DejaVu Sans Mono")
    return y_top - h


def data_design():
    fig, ax = canvas(7.2, 6.4)
    ax.text(3.6, 6.2, "One row per subject (RID) unless stated; lines join tables that share RID or image_id.",
            ha="center", fontsize=FS - 0.5, color=INK2, style="italic")
    xs = [1.15, 3.6, 6.05]
    tops = [5.85, 3.85, 1.85]
    cols = [
        [("cn_cohort", ["RID (key)", "image_id", "age, sex, scan_date"], DONE_FILL),
         ("cn_splits", ["RID (key)", "split: train/val/test"], DONE_FILL),
         ("cn_train_scans", ["RID", "image_id (all CN visits)", "age"], DONE_FILL)],
        [("scan_qc_final", ["cohort, RID, split", "image_id (QC-passed)", "status"], NEUTRAL),
         ("voxels_4mm_index", ["image_id (key)", "cohort, split, RID", "row -> voxels_4mm.npy"], NEUTRAL),
         ("bag_table", ["RID (key), image_id", "age, brain_age", "bag_raw, bag, label"], DATA_FILL)],
        [("mci_cohort", ["RID (key)", "image_id", "age, sex", "label (1 = progressor)"], DATA_FILL),
         ("mci_splits", ["RID (key)", "split: train/val/test"], DATA_FILL),
         ("mci_voxels_4mm_index", ["RID (key), image_id", "split", "row -> E1-E5 features"], DATA_FILL)],
    ]
    bottoms = {}
    for c, (x, col) in enumerate(zip(xs, cols)):
        for r, (title, fields, fill) in enumerate(col):
            bottoms[(c, r)] = table(ax, x, tops[r], title, fields, fill)
    line = dict(color=MUTED, lw=0.9)
    for c, x in enumerate(xs):                       # vertical links within each column
        for r in (0, 1):
            ax.plot([x, x], [bottoms[(c, r)], tops[r + 1]], **line)
    y = tops[0] - 0.55                               # horizontal links, top row
    ax.plot([xs[0] + 1.1, xs[1] - 1.1], [y, y], **line)
    ax.plot([xs[1] + 1.1, xs[2] - 1.1], [y, y], **line)
    y = tops[2] - 0.45                               # bag_table <-> mci_voxels_4mm_index
    ax.plot([xs[1] + 1.1, xs[2] - 1.1], [y, y], **line)
    save(fig, "data_design.png")


# ---------------------------------------------------------------- 4. freeze workflow

def freeze_workflow():
    fig, ax = canvas(7.2, 2.3)
    steps = ["Fix the rules" + NL + "(decision log," + NL + "before training)",
             "Train + tune" + NL + "5-fold CV," + NL + "training data only",
             "Check on" + NL + "validation" + NL + "(no test data)",
             "Freeze" + NL + "commit choice" + NL + "+ file hashes",
             "Test set once" + NL + "--confirm-test;" + NL + "2nd run refused"]
    w, h, y = 1.26, 0.95, 1.55
    xs = [0.7 + i * 1.45 for i in range(5)]
    for i, (x, t) in enumerate(zip(xs, steps)):
        box(ax, x, y, w, h, t, fill=DONE_FILL if i < 4 else DATA_FILL, size=FS - 0.5)
        if i:
            arrow(ax, xs[i - 1] + w / 2, y, x - w / 2, y)
    ax.annotate("", xy=(xs[1], y - h / 2), xytext=(xs[2], y - h / 2),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, connectionstyle="arc3,rad=-0.5"))
    ax.text(xs[1] + 0.7, 0.22, "change a setting: tune again on training data", ha="center",
            fontsize=FS - 1, color=INK2)
    ax.text(xs[3] + 0.7, 0.22, "hashes must match; validation must reproduce", ha="center",
            fontsize=FS - 1, color=INK2)
    save(fig, "freeze_workflow.png")


# ---------------------------------------------------------------- 5. gantt

def gantt():
    fyp1 = [("Literature review and proposal", 0, 8.5),
            ("ADNI access, repo, environment", 6, 10.5),
            ("Cohort definition and splits", 10.5, 11.0),
            ("MRI preprocessing and QC", 10.5, 12.0),
            ("Brain-age models + correction", 12.0, 12.8),
            ("BAG and group test", 12.6, 12.9),
            ("E1 and E2, frozen test", 12.9, 13.1),
            ("FYP-I report and presentation", 12.9, 15.0)]
    m0, mon = 18, 4.3            # x axis in weeks; FYP-II month k starts at m0 + k * mon
    fyp2 = [("E3 and E4 (BAG experiments)", 0, 1.5),
            ("E5 clinical fusion (optional)", 1.2, 2.2),
            ("Statistics; BAG test on all MCI", 2.0, 3.0),
            ("Explainability (XAI)", 1.5, 3.5),
            ("Calibration", 2.5, 3.5),
            ("Integration, reproducibility", 3.0, 4.0),
            ("Thesis, slides and defence", 3.5, 5.0)]
    rows = [(n, s, e, True) for n, s, e in fyp1] + [(n, m0 + s * mon, m0 + e * mon, False) for n, s, e in fyp2]
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0.33, 0.08, 0.66, 0.84])
    for i, (name, s, e, done) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, max(e - s, 0.3), left=s, height=0.55, color=BLUE if done else SURFACE,
                edgecolor=BLUE if done else ORANGE, lw=1.1, hatch=None if done else "////")
        ax.text(-0.6, y, name, ha="right", va="center", fontsize=FS - 0.5, color=INK)
    ax.axvline(m0 - 1, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ticks = [0, 4.3, 8.7, 13.0] + [m0 + k * mon for k in range(5)]
    labels = ["Sep", "Oct", "Nov", "Dec"] + [f"M{k + 1}" for k in range(5)]
    ax.set_xticks(ticks, labels, fontsize=FS - 0.5, color=INK2)
    ax.set_yticks([])
    ax.set_xlim(0, m0 + 5 * mon)
    ax.set_ylim(-0.7, len(rows) + 0.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(7.5, len(rows) - 0.2, "FYP-I (2026, done)", fontsize=FS, color=INK, fontweight="bold", ha="center")
    ax.text(m0 + 2.5 * mon, len(rows) - 0.2, "FYP-II (planned; dates TBC)", fontsize=FS, color=INK,
            fontweight="bold", ha="center")
    save(fig, "gantt.png")


def main():
    print("Writing design diagrams to docs/figures:")
    pipeline_overview()
    architecture()
    data_design()
    freeze_workflow()
    gantt()


if __name__ == "__main__":
    main()
