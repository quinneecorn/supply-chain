"""
Oversample minority classes (1, 2) by duplicating rows — safe text augmentation.

Usage:
  python augment_minority.py
  # writes data/augmented_train.csv — set CSV_PATH in .env and retrain
"""

from __future__ import annotations

import csv
from pathlib import Path

from config import CSV_LABEL_COL, CSV_PATH, CSV_SENTENCE_COL, ENTITY_FROM, ENTITY_TO

# How many extra copies per original row (0 = no extra; 2 = 3x total)
COPIES_CLASS_1 = 2
COPIES_CLASS_2 = 4
OUTPUT_PATH = Path("data/augmented_train.csv")


def _valid_row(row: dict) -> bool:
    m = str(row.get(CSV_SENTENCE_COL) or "")
    rid = row.get(CSV_LABEL_COL)
    if not m or ENTITY_FROM not in m or ENTITY_TO not in m:
        return False
    if rid is None or str(rid).strip() in ("", "None", "null"):
        return False
    try:
        int(float(rid))
    except (TypeError, ValueError):
        return False
    return True


def main() -> None:
    src = Path(CSV_PATH)
    if not src.is_file():
        raise FileNotFoundError(src)

    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [r for r in reader if _valid_row(r)]

    out_rows: list[dict] = []
    stats = {1: 0, 2: 0, "other": 0}

    for row in rows:
        out_rows.append(row)
        try:
            rid = int(float(row[CSV_LABEL_COL]))
        except (TypeError, ValueError):
            continue
        copies = 0
        if rid == 1:
            copies = COPIES_CLASS_1
            stats[1] += copies
        elif rid == 2:
            copies = COPIES_CLASS_2
            stats[2] += copies
        else:
            stats["other"] += 0
        for _ in range(copies):
            out_rows.append(dict(row))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Source valid rows: {len(rows)}")
    print(f"Output rows: {len(out_rows)} (+{len(out_rows) - len(rows)} duplicates)")
    print(f"Extra copies — class 1: {stats[1]}, class 2: {stats[2]}")
    print(f"Written: {OUTPUT_PATH.resolve()}")
    print("Set CSV_PATH=data/augmented_train.csv in .env then: python train.py")


if __name__ == "__main__":
    main()
