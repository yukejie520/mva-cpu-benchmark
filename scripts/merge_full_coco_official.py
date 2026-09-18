"""Merge per-run official COCOeval CSVs without rerunning inference."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["YOLO11n", "YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "RT-DETR-l", "RT-DETR-x"]
FIELDS = ["name", "kind", "images", "n_pred", "map50_95", "map50", "map75", "mar100"]


def main() -> None:
    paths = [ROOT / "results" / "full_coco_map_official.csv",
             ROOT / "results" / "full_coco_map_official_rest.csv",
             ROOT / "results" / "full_coco_map_official_missing.csv",
             ROOT / "results" / "full_coco_map_official_rtdetrx.csv"]
    rows = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows[row["name"]] = row
    if set(rows) != set(ORDER) or any(int(rows[name]["images"]) != 5000 for name in ORDER):
        raise RuntimeError(f"expected all seven 5,000-image rows, got {sorted(rows)}")
    out = ROOT / "results" / "full_coco_map_official.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[name] for name in ORDER)
    print(out)


if __name__ == "__main__":
    main()
