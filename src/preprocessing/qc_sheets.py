"""Put the QC pictures of many scans onto contact sheets for visual review.

Run from the repo root:  python -m src.preprocessing.qc_sheets [--per-sheet 10]

Takes every successfully preprocessed scan from outputs/preprocessing/qc_metrics.csv, sorted by
QC score (lowest first, so the most doubtful scans are seen first), and pastes their QC pictures
(three middle slices with the template brain outline in red) onto sheets of --per-sheet scans.
Writes outputs/preprocessing/qc_sheets/sheet_NNN.png and sheet_index.csv (sheet, position,
image ID, score), all local only. Sheets are rebuilt from scratch on each run.
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

QC_DIR = Path("outputs/preprocessing")
OUT = QC_DIR / "qc_sheets"
TILE_WIDTH = 640  # each QC picture is scaled to this width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sheet", type=int, default=10)
    parser.add_argument("--new-only", action="store_true",
                        help="leave out scans already in visual_review.csv")
    args = parser.parse_args()

    metrics = pd.read_csv(QC_DIR / "qc_metrics.csv")
    ok = metrics[metrics["status"] == "ok"].sort_values("template_correlation")
    review = QC_DIR / "visual_review.csv"
    if args.new_only and review.exists():
        done = set(pd.read_csv(review)["image_id"].astype(str))
        ok = ok[~ok["image_id"].astype(str).isin(done)]
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("sheet_*.png"):
        old.unlink()

    index = []
    ids = ok["image_id"].tolist()
    scores = dict(zip(ok["image_id"], ok["template_correlation"]))
    for sheet, start in enumerate(range(0, len(ids), args.per_sheet), 1):
        batch = ids[start:start + args.per_sheet]
        tiles = []
        for image_id in batch:
            tile = Image.open(QC_DIR / "qc" / f"{image_id}.png").convert("RGB")
            height = round(tile.height * TILE_WIDTH / tile.width)
            tiles.append(tile.resize((TILE_WIDTH, height)))
        columns = 2
        rows = -(-len(tiles) // columns)
        width, height = TILE_WIDTH, tiles[0].height
        page = Image.new("RGB", (columns * width, rows * height), "white")
        for position, tile in enumerate(tiles):
            page.paste(tile, ((position % columns) * width, (position // columns) * height))
            index.append({"sheet": sheet, "position": position + 1,
                          "image_id": batch[position],
                          "template_correlation": scores[batch[position]]})
        page.save(OUT / f"sheet_{sheet:03d}.png")

    pd.DataFrame(index).to_csv(OUT / "sheet_index.csv", index=False)
    print("scans:", len(ids), "| sheets:", -(-len(ids) // args.per_sheet), "in", OUT)


if __name__ == "__main__":
    main()
