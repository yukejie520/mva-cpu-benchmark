"""Export the small, path-sanitized full-val COCO release package."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "coco_val2017"
RESULTS = ROOT / "results"
OUT = ROOT / "release_data"


def main() -> None:
    manifest = json.loads((DATA / "val2017_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("annotations_source"):
        raise RuntimeError("annotation provenance is missing from val2017_manifest.json")
    out_manifest = {
        "image_archive": "COCO val2017.zip",
        "image_sha256": manifest["sha256"],
        "image_count": manifest["count"],
        "annotation_file": "instances_val2017.json",
        "annotation_sha256": manifest["annotations_sha256"],
        "annotation_count": 36781,
        "non_crowd_target_count": 36335,
        "evaluation_command": "python scripts/evaluate_full_coco_official.py",
        "evaluation": {
            "implementation": "pycocotools COCOeval 2.0.11",
            "input_size": 640,
            "confidence_threshold": 0.001,
            "iou_thresholds": "0.50:0.05:0.95",
            "threads": 16,
            "runtime": "ONNX Runtime 1.28.0 CPUExecutionProvider",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "full_coco_manifest.json").write_text(
        json.dumps(out_manifest, indent=2) + "\n", encoding="utf-8"
    )
    names = "\n".join(manifest["images"]) + "\n"
    (OUT / "full_coco_image_list.txt").write_text(names, encoding="utf-8")
    # Only export the standard COCOeval table.  The older NumPy table is kept
    # as an audit artifact and must not silently enter the public release.
    source = RESULTS / "full_coco_map_official.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    expected_names = {"YOLO11n", "YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "RT-DETR-l", "RT-DETR-x"}
    if (len(rows) != 7 or {int(r["images"]) for r in rows} != {5000}
            or {r["name"] for r in rows} != expected_names):
        raise RuntimeError("full_coco_map_official.csv must contain seven 5,000-image rows")
    expected = {"name", "map50_95", "map50", "map75", "mar100"}
    if not expected.issubset(rows[0]):
        raise RuntimeError("official table is missing COCOeval summary fields")
    (OUT / "full_coco_map.csv").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Exported {len(rows)} full-val rows and {len(manifest['images'])} image names")


if __name__ == "__main__":
    main()
