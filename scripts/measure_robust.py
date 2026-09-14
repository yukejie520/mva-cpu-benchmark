"""多轮稳健延迟测量：每模型独立 R 轮，报「中位数的中位数」+ 跨轮区间。

为什么需要（notes/04_w3c_full_results.md）：消费级笔记本 CPU 睿频/温度使单窗口绝对延迟
跨时段漂移 ~30%（YOLOv8n 探针 31.8ms vs 正式 21.8ms）。单次测虽用 240 样本中位，
仍受所在"窗口"整体快慢影响。多轮独立重测（每轮含各自 warmup），跨轮中位数再取中位，
并报跨轮 min~max 区间，供论文写 ±区间与可复现性。

用法：
    python scripts/measure_robust.py --rounds 3 --imgs data/coco128/coco128/images/train2017 \
        --out results/latency_robust.csv
    # 每模型：warmup 20 × 1 轮内预热 + 8 图 × 30 reps → 240 样本/轮，R 轮独立
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_common import IMGSZ, build_session, list_images, preprocess  # noqa: E402
from measure import REGISTRY, _measure  # noqa: E402


def round_stats(medians: list[float]) -> tuple[float, float, float]:
    """跨轮聚合：返回 (中位数之中位数, 跨轮最小值, 跨轮最大值)。"""
    if not medians:
        raise ValueError("medians 不能为空")
    return statistics.median(medians), min(medians), max(medians)


CSV_FIELDS = ["name", "family", "params_M", "gflops",
              "fwd_median_ms", "fwd_lo_ms", "fwd_hi_ms",
              "e2e_median_ms", "e2e_lo_ms", "e2e_hi_ms", "fps", "rounds"]


def _append_csv(rows: list[dict], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(out_path).exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--models", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = list_images(args.imgs, args.n_imgs)
    if not paths:
        raise FileNotFoundError(f"no jpg under {args.imgs}")
    inputs = [preprocess(str(p), IMGSZ) for p in paths]

    want = set(args.models.split(",")) if args.models else None
    rows = []
    for cfg in REGISTRY:
        if want and cfg["name"] not in want:
            continue
        if not Path(cfg["onnx"]).exists():
            print(f"[skip] {cfg['name']}: {cfg['onnx']} 不存在")
            continue
        print(f"[robust] {cfg['name']} rounds={args.rounds} "
              f"(n_imgs={len(paths)}, warmup={args.warmup}, reps={args.reps})", flush=True)
        sess = build_session(cfg["onnx"])  # 16 线程 + ORT 优化，与单窗口口径一致
        fwd_meds, e2e_meds = [], []
        for r in range(args.rounds):
            row = _measure(sess, cfg, inputs, args.warmup, args.reps)
            fwd_meds.append(row["fwd_median_ms"])
            e2e_meds.append(row["e2e_median_ms"])
            print(f"  round {r + 1}/{args.rounds}: fwd={row['fwd_median_ms']:.1f}ms "
                  f"e2e={row['e2e_median_ms']:.1f}ms", flush=True)
        fm, flo, fhi = round_stats(fwd_meds)
        em, elo, ehi = round_stats(e2e_meds)
        rows.append({
            "name": cfg["name"], "family": cfg["family"],
            "params_M": row["params_M"], "gflops": row["gflops"],
            "fwd_median_ms": round(fm, 3), "fwd_lo_ms": round(flo, 3), "fwd_hi_ms": round(fhi, 3),
            "e2e_median_ms": round(em, 3), "e2e_lo_ms": round(elo, 3), "e2e_hi_ms": round(ehi, 3),
            "fps": round(1000.0 / em, 2), "rounds": args.rounds,
        })
        print(f"  -> e2e 中位之中位={em:.1f}ms (跨轮 {elo:.1f}~{ehi:.1f})  fps={1000.0/em:.1f}")

    print("| name | e2e median_ms | e2e range | fps |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['e2e_median_ms']:.1f} | "
              f"{r['e2e_lo_ms']:.1f}~{r['e2e_hi_ms']:.1f} | {r['fps']:.1f} |")
    if args.out:
        _append_csv(rows, args.out)
        print(f"[csv] appended {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
