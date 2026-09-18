"""Run the seven FP32 ONNX models through the official pycocotools COCOeval.

This is deliberately separate from ``evaluate_full_coco.py``.  The latter is
kept as a lightweight NumPy exploratory evaluator; this entry point uses the
standard COCOeval implementation and writes raw detections.  The resulting
numbers become submission-grade only after the annotation JSON is verified
against the official COCO archive (the current local copy is a documented
third-party mirror).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["YOLO11n", "YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "RT-DETR-l", "RT-DETR-x"]
FILES = {
    "YOLO11n": "yolo11n.onnx", "YOLOv8n": "yolov8n.onnx",
    "YOLOv8s": "yolov8s.onnx", "YOLOv8m": "yolov8m.onnx",
    "YOLOv8l": "yolov8l.onnx", "RT-DETR-l": "rtdetr-l.onnx",
    "RT-DETR-x": "rtdetr-x.onnx",
}


def run_model(name: str, images: Path, annotations: Path, pred_dir: Path, threads: int,
              resume: bool = False, max_images: int = 0,
              shard_index: int = 0, shard_count: int = 1,
              prediction_suffix: str = "", model_path: Path | None = None) -> Path:
    """Generate one COCO detection JSON using the existing decode pipeline."""
    # Import lazily so ``--evaluate-only`` still works if inference dependencies
    # are unavailable on a review machine that already has prediction JSONs.
    from eval_map import _Subset, decode_rtdetr, decode_yolo, preprocess_meta
    from eval_common import build_session
    from coco_eval import COCO_CAT_IDS

    model = model_path if model_path is not None else ROOT / "data" / FILES[name]
    if not model.exists():
        raise FileNotFoundError(model)
    kind = "yolo" if name.startswith("YOLO") else "rtdetr"
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    subset = _Subset(str(annotations))
    if shard_count > 1:
        subset.images = subset.images[shard_index::shard_count]
    if max_images:
        subset.images = subset.images[:max_images]
    sess = build_session(str(model), threads=threads)
    inp_name = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    pred_dir.mkdir(parents=True, exist_ok=True)
    partial = pred_dir / f"{name}_predictions{prediction_suffix}.partial.json"
    detections: list[dict] = []
    start_index = 0
    if resume and partial.exists():
        state = json.loads(partial.read_text(encoding="utf-8"))
        start_index = int(state["processed_images"])
        detections = state["detections"]
        if not 0 <= start_index <= len(subset.images):
            raise ValueError(f"invalid checkpoint: {partial}")
        print(f"[{name}] resuming at image {start_index}/{len(subset.images)}", flush=True)
    start = time.perf_counter()
    for index, im in enumerate(subset.images[start_index:], start_index + 1):
        path = images / im["file_name"]
        if not path.exists():
            raise FileNotFoundError(path)
        x, meta = preprocess_meta(str(path))
        raw = sess.run(outs, {inp_name: x})[0]
        if kind == "yolo":
            boxes, scores, classes = decode_yolo(raw, meta, 0.001, 0.7)
        else:
            boxes, scores, classes = decode_rtdetr(raw, meta, 0.001)
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "image_id": int(im["id"]),
                "category_id": int(COCO_CAT_IDS[int(cls)]),
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score),
            })
        if index % 500 == 0 or index == len(subset.images):
            label = f"{name}{prediction_suffix}"
            print(f"[{label}] {index}/{len(subset.images)} images", flush=True)
            partial.write_text(json.dumps({"processed_images": index,
                                           "detections": detections},
                                          separators=(",", ":")), encoding="utf-8")
    out = pred_dir / f"{name}_predictions{prediction_suffix}.json"
    out.write_text(json.dumps(detections, separators=(",", ":")), encoding="utf-8")
    try:
        partial.unlink(missing_ok=True)
    except PermissionError:
        # Windows can briefly retain a read handle after a large checkpoint
        # write. The completed JSON is authoritative, so leave the checkpoint
        # in place and keep the run resumable instead of failing at cleanup.
        print(f"[{name}] checkpoint retained because it is still in use: {partial}", flush=True)
    print(f"[{name}] wrote {len(detections)} detections in {time.perf_counter()-start:.1f}s -> {out}")
    return out


def evaluate_one(coco_gt: COCO, pred_path: Path) -> dict:
    """Evaluate one JSON with the official COCOeval defaults."""
    coco_dt = coco_gt.loadRes(str(pred_path))
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = sorted(coco_gt.getImgIds())
    ev.params.catIds = sorted(coco_gt.getCatIds())
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats
    name = pred_path.name.removesuffix("_predictions.json")
    return {
        "name": name,
        "kind": "yolo" if name.startswith("YOLO") else "rtdetr",
        "images": len(ev.params.imgIds),
        "n_pred": len(json.loads(pred_path.read_text(encoding="utf-8"))),
        "map50_95": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "mar100": float(stats[8]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "data/coco_val2017/val2017"))
    ap.add_argument("--annotations", default=str(ROOT / "data/coco_val2017/annotations/instances_val2017.json"))
    ap.add_argument("--pred-dir", default=str(ROOT / "results/official_predictions"))
    ap.add_argument("--out", default=str(ROOT / "results/full_coco_map_official.csv"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--model-spec", action="append", default=[], metavar="NAME=PATH",
                    help="override/add an ONNX artifact, repeatable; relative paths use the repo root")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--reuse-predictions", action="store_true",
                    help="reuse an existing per-model JSON and continue missing models")
    ap.add_argument("--resume-predictions", action="store_true",
                    help="resume a .partial.json checkpoint written every 500 images")
    ap.add_argument("--max-images", type=int, default=0,
                    help="development smoke-test limit; leave at 0 for the full 5000 images")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="zero-based inference shard index")
    ap.add_argument("--shard-count", type=int, default=1,
                    help="number of disjoint inference shards")
    ap.add_argument("--prediction-suffix", default="",
                    help="suffix inserted before _predictions.json for shard outputs")
    ap.add_argument("--inference-only", action="store_true",
                    help="write prediction JSON without running COCOeval")
    args = ap.parse_args()
    model_paths: dict[str, Path] = {}
    for spec in args.model_spec:
        if "=" not in spec:
            raise ValueError(f"invalid --model-spec {spec!r}; expected NAME=PATH")
        name, raw_path = spec.split("=", 1)
        name, raw_path = name.strip(), raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"invalid --model-spec {spec!r}; expected NAME=PATH")
        path = Path(raw_path)
        model_paths[name] = path if path.is_absolute() else ROOT / path
    images, annotations, pred_dir = map(Path, (args.images, args.annotations, args.pred_dir))
    if not images.exists() or not annotations.exists():
        raise FileNotFoundError("Prepare COCO val2017 images and annotations first")
    coco_gt = None if args.inference_only else COCO(str(annotations))
    selected = [x.strip() for x in args.models.split(",") if x.strip()]
    unknown = sorted(set(selected) - (set(MODELS) | set(model_paths)))
    if unknown:
        raise ValueError(f"unknown model(s): {unknown}")
    rows = []
    for name in selected:
        pred = pred_dir / f"{name}_predictions{args.prediction_suffix}.json"
        if not args.evaluate_only and not (args.reuse_predictions and pred.exists()):
            pred = run_model(name, images, annotations, pred_dir, args.threads,
                             resume=args.resume_predictions, max_images=args.max_images,
                             shard_index=args.shard_index, shard_count=args.shard_count,
                             prediction_suffix=args.prediction_suffix,
                             model_path=model_paths.get(name))
        if not pred.exists():
            raise FileNotFoundError(pred)
        if not args.inference_only:
            rows.append(evaluate_one(coco_gt, pred))
    if args.inference_only:
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "kind", "images", "n_pred", "map50_95", "map50", "map75", "mar100"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
