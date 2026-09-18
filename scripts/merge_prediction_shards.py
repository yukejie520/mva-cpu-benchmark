"""Merge disjoint official-inference JSON shards into one COCO result file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard-count", type=int, required=True)
    ap.add_argument("--pred-dir", default="results/official_predictions")
    ap.add_argument("--annotations", default="data/coco_val2017/annotations/instances_val2017.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.shard_count < 2:
        raise ValueError("shard-count must be >= 2")
    pred_dir = Path(args.pred_dir)
    rows: list[dict] = []
    seen: set[int] = set()
    for index in range(args.shard_count):
        path = pred_dir / f"{args.model}_predictions.part{index}of{args.shard_count}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        shard = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(shard, list):
            raise ValueError(f"expected a JSON list: {path}")
        shard_ids = {int(row["image_id"]) for row in shard}
        duplicate = seen.intersection(shard_ids)
        if duplicate:
            raise ValueError(f"duplicate image IDs across shards: {sorted(duplicate)[:5]}")
        seen.update(shard_ids)
        rows.extend(shard)

    anns = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    expected = {int(im["id"]) for im in anns["images"]}
    missing = expected - seen
    extra = seen - expected
    if missing or extra:
        raise ValueError(f"shard coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    out = Path(args.out) if args.out else pred_dir / f"{args.model}_predictions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    print(f"merged {args.shard_count} shards, {len(rows)} detections, {len(seen)} images -> {out}")


if __name__ == "__main__":
    main()
