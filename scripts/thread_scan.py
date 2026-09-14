"""测量协议诊断：ONNX Runtime CPU 线程扫描（CLI 薄壳，逻辑在 eval_common）。

目的：正式测延迟前弄清「开几个线程」最稳。已扫描结论：8-16 甜点，
24 开满 +28% 变差 → 用户已拍板全篇用 16 线程（eval_common.DEFAULT_THREADS）。

用法：
    python scripts/thread_scan.py --model data/yolov8n.onnx --threads 1,4,8,16,24 \
        --imgs data/coco128/... --n-imgs 8 --warmup 8 --reps 3
"""
from __future__ import annotations

import argparse

from eval_common import aggregate, build_session, letterbox, list_images, preprocess, measure_forward  # noqa: F401  (letterbox 由 preprocess 使用)


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX CPU thread-count latency scan")
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", default="1,4,8,16,24")
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    paths = list_images(args.imgs, args.n_imgs)
    if not paths:
        raise FileNotFoundError(f"no jpg under {args.imgs}")
    inputs = [preprocess(str(p), args.imgsz) for p in paths]

    print(f"model={args.model} | imgs={len(paths)} warmup={args.warmup} reps={args.reps} imgsz={args.imgsz}")
    print("| threads | median_ms | mean_ms | min_ms | p90_ms | std_ms | fps |")
    print("|---|---|---|---|---|---|---|")
    for t in [int(x) for x in args.threads.split(",") if x.strip()]:
        sess = build_session(args.model, threads=t)
        times = measure_forward(sess, inputs, args.warmup, args.reps)
        a = aggregate(times)
        print(f"| {t} | {a['median_ms']:.1f} | {a['mean_ms']:.1f} | {a['min_ms']:.1f} "
              f"| {a['p90_ms']:.1f} | {a['std_ms']:.1f} | {a['fps']:.1f} |", flush=True)


if __name__ == "__main__":
    main()
