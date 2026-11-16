"""Record visual-review verdicts for whole QC sheets (local review table, never committed).

Run from the repo root, for example:
    python -m src.preprocessing.record_review --sheets 5 6 7
    python -m src.preprocessing.record_review --sheets 8 --set 12345=retry:"pitched forward"

Every scan on the listed sheets is marked "ok" unless --set gives another verdict
(retry = redo registration, quality = image artefact, fail = cannot be used).
Verdicts are kept in outputs/preprocessing/visual_review.csv; a later verdict replaces an earlier one.
"""
import argparse
from pathlib import Path

import pandas as pd

QC_DIR = Path("outputs/preprocessing")
REVIEW = QC_DIR / "visual_review.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheets", type=int, nargs="+", required=True)
    parser.add_argument("--set", nargs="*", default=[], help='image_id=verdict:"note"')
    args = parser.parse_args()

    index = pd.read_csv(QC_DIR / "qc_sheets" / "sheet_index.csv")
    ids = index[index["sheet"].isin(args.sheets)]["image_id"].tolist()
    special = {}
    for item in args.set:
        image_id, rest = item.split("=", 1)
        verdict, _, note = rest.partition(":")
        special[int(image_id)] = (verdict, note)
    new = pd.DataFrame([{"image_id": i, "verdict": special.get(i, ("ok", ""))[0],
                         "note": special.get(i, ("ok", ""))[1]} for i in ids])
    old = pd.read_csv(REVIEW) if REVIEW.exists() else new.iloc[0:0]
    table = pd.concat([old[~old["image_id"].isin(new["image_id"])], new])
    table.to_csv(REVIEW, index=False)
    print("recorded:", len(new), "| total reviewed:", len(table),
          "|", table["verdict"].value_counts().to_dict())


if __name__ == "__main__":
    main()
