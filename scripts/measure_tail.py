"""逐样本尾部延迟采集：每模型同一连续窗 20 预热 + 240 计时样本，落盘供 max/p95/p99。

背景（R11 补测，2026-09-09）：latency_canonical7.csv 只落了「跨轮中位数的中位数 +
跨轮 min~max」，240 个逐样本延迟从未存档 → 硬帧限门控要的尾部统计（max/p95/p99）
从现成聚合算不出来。本脚本在**新窗口**按与 canonical7 相同的协议重采逐样本 e2e：
- 台架/口径同一：ORT 1.28.0 CPU、16 线程、640 letterbox、端到端计时
  （YOLO = forward + Python NMS；RT-DETR = forward 含解码）。
- 顺序同一：REGISTRY 注册表顺序（YOLO11n 在尾）；输入同一 8 图 × 30 reps = 240 样本。
- 区别仅在新窗口（2026-09-09）：绝对中位与 canonical7 不必相等（跨日漂移本就如实报告），
  尾部形状才是本脚本交付物。

用法：
    # 真实跑（后台，~7 分钟），freq_log_stream 并行记窗内频率：
    python scripts/measure_tail.py --imgs data/coco128/coco128/images/train2017 \
        --out results/tail_samples.csv
    # 冒烟（5 样本/模型，验证管路）：
    python scripts/measure_tail.py --imgs data/coco128/coco128/images/train2017 \
        --n-imgs 1 --reps 5 --warmup 5 --out results/_tail_smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_common import IMGSZ, build_session, list_images, preprocess  # noqa: E402
from measure import REGISTRY  # noqa: E402  (模型注册表/顺序唯一来源)
from yolo_post import yolov8_nms  # noqa: E402


def tail_stats(samples_ms: list[float]) -> dict:
    """逐样本 e2e(ms) -> {n, median, p95, p99, max, mean, std}。

    百分位用线性插值（np.percentile 默认，与业界 p95/p99 惯例一致）。
    """
    a = np.array(samples_ms, dtype=np.float64)
    if a.size == 0:
        raise ValueError("samples must be non-empty")
    return {
        "n": int(a.size),
        "median_ms": round(float(np.median(a)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
        "max_ms": round(float(a.max()), 3),
        "mean_ms": round(float(a.mean()), 3),
        "std_ms": round(float(a.std()), 3),
    }


def measure_one(sess, cfg: dict, inputs: list[np.ndarray],
                warmup: int, reps: int) -> tuple[list[float], list[float]]:
    """对单模型采 e2e/fwd 逐样本(ms)。计时区与 measure._measure 完全一致。"""
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    kind = cfg["kind"]
    for _ in range(warmup):  # 预热（第一张，含 yolo 后处理，不计时）
        out = sess.run(outs, {inp: inputs[0]})
        if kind == "yolo":
            yolov8_nms(out[0])
    fwd_t, e2e_t = [], []
    for x in inputs:
        for _ in range(reps):
            t0 = time.perf_counter()
            out = sess.run(outs, {inp: x})
            t1 = time.perf_counter()
            fwd_t.append((t1 - t0) * 1e3)
            if kind == "yolo":
                yolov8_nms(out[0])
                e2e_t.append((time.perf_counter() - t0) * 1e3)
            else:
                e2e_t.append((t1 - t0) * 1e3)  # RT-DETR 端到端即前向
    return e2e_t, fwd_t


def _write(out_path: str, rows: list[dict], fields: list[str]) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(out_path).exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--models", default=None)
    ap.add_argument("--out", default=None, help="逐样本 CSV（追加）")
    args = ap.parse_args()

    paths = list_images(args.imgs, args.n_imgs)
    if not paths:
        raise FileNotFoundError(f"no jpg under {args.imgs}")
    inputs = [preprocess(str(p), IMGSZ) for p in paths]

    want = set(args.models.split(",")) if args.models else None
    for cfg in REGISTRY:
        if want and cfg["name"] not in want:
            continue
        onnx = cfg["onnx"]
        if not Path(onnx).exists():
            print(f"[skip] {cfg['name']}: {onnx} 不存在", flush=True)
            continue
        print(f"[tail] {cfg['name']} ({cfg['family']}, n_imgs={len(paths)}, "
              f"warmup={args.warmup}, reps={args.reps})", flush=True)
        sess = build_session(onnx)
        e2e_t, fwd_t = measure_one(sess, cfg, inputs, args.warmup, args.reps)
        st = tail_stats(e2e_t)
        print(f"  -> e2e median={st['median_ms']}ms  p95={st['p95_ms']}  "
              f"p99={st['p99_ms']}  max={st['max_ms']}  (n={st['n']})", flush=True)
        if args.out:
            _write(args.out, [{"name": cfg["name"], "sample": i, "e2e_ms": round(v, 3)}
                              for i, v in enumerate(e2e_t)],
                   ["name", "sample", "e2e_ms"])
    if args.out:
        print(f"[csv] appended per-sample rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
