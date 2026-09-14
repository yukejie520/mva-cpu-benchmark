"""统一效率测量：一次跑出 参数量/FLOPs/前向延迟/端到端延迟（16 线程）。

协议（账本 §0，2026-09-05 拍板）：
- 台架：本地 i7-14650HX；ORT intra_op=16 线程
- 输入：真实图 letterbox 640；warmup 20；测次 reps（大模型按时长减）
- 口径：端到端为主（YOLO=forward+Python NMS；RT-DETR=forward 含解码免 NMS）+ forward 副列
- 延迟取中位（aggregate）

探针 / 全量共用；模型按 REGISTRY 声明，--models 可过滤。
用法：
    python scripts/measure.py --imgs data/coco128/coco128/images/train2017 \
        --n-imgs 8 --warmup 20 --reps 30 --out results/probe_raw.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from eval_common import IMGSZ, aggregate, build_session, list_images, preprocess  # noqa: E402
from onnx_metrics import onnx_gflops, onnx_params_m  # noqa: E402
from yolo_post import yolov8_nms  # noqa: E402

# 模型注册表：名字/家族/onnx 路径/类型。YOLO 族有 NMS 后处理，RT-DETR 端到端。
REGISTRY = [
    {"name": "YOLOv8n", "family": "CNN", "onnx": "data/yolov8n.onnx", "kind": "yolo"},
    {"name": "YOLOv8s", "family": "CNN", "onnx": "data/yolov8s.onnx", "kind": "yolo"},
    {"name": "YOLOv8m", "family": "CNN", "onnx": "data/yolov8m.onnx", "kind": "yolo"},
    {"name": "YOLOv8l", "family": "CNN", "onnx": "data/yolov8l.onnx", "kind": "yolo"},
    {"name": "RT-DETR-l", "family": "Transformer", "onnx": "data/rtdetr-l.onnx", "kind": "rtdetr"},
    {"name": "RT-DETR-x", "family": "Transformer", "onnx": "data/rtdetr-x.onnx", "kind": "rtdetr"},
    # YOLO11n（2026-09-06 补）：近年轻量 CNN，加入最终 7 模型集做 LAE/Pareto 秩稳健辩护。
    # 放注册表末尾避免打乱既有 6 模型的顺序/编号；kind=yolo 走与 YOLOv8 相同的 decode+NMS 口径。
    {"name": "YOLO11n", "family": "CNN", "onnx": "data/yolo11n.onnx", "kind": "yolo"},
]


def _measure(sess: ort.InferenceSession, cfg: dict, inputs: list[np.ndarray],
             warmup: int, reps: int) -> dict:
    """对单个 session 测前向 + 端到端延迟。返回含 params/gflops 的完整行。"""
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    kind = cfg["kind"]
    # 预热（第一张，含 yolo 后处理，不计时）
    for _ in range(warmup):
        out = sess.run(outs, {inp: inputs[0]})
        if kind == "yolo":
            yolov8_nms(out[0])
    fwd_times, e2e_times = [], []
    for x in inputs:
        for _ in range(reps):
            t0 = time.perf_counter()
            out = sess.run(outs, {inp: x})
            t1 = time.perf_counter()
            fwd_times.append((t1 - t0) * 1e3)
            if kind == "yolo":
                yolov8_nms(out[0])
                t2 = time.perf_counter()
                e2e_times.append((t2 - t0) * 1e3)
            else:
                e2e_times.append((t1 - t0) * 1e3)  # RT-DETR 端到端即前向
    af, ae = aggregate(fwd_times), aggregate(e2e_times)
    return {
        "name": cfg["name"], "family": cfg["family"], "kind": kind,
        "params_M": onnx_params_m(cfg["onnx"]),
        "gflops": onnx_gflops(cfg["onnx"], IMGSZ)["gflops"],
        "fwd_median_ms": af["median_ms"], "fwd_std_ms": af["std_ms"],
        "e2e_median_ms": ae["median_ms"], "e2e_std_ms": ae["std_ms"],
        "fps": ae["fps"],
    }


def _print_table(rows: list[dict]) -> None:
    print("| name | family | params_M | gflops | fwd_ms | e2e_ms | fps |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['family']} | {r['params_M']:.1f} | {r['gflops']:.1f} "
              f"| {r['fwd_median_ms']:.1f} | {r['e2e_median_ms']:.1f} | {r['fps']:.1f} |")


def _append_csv(rows: list[dict], out_path: str) -> None:
    fields = ["name", "family", "params_M", "gflops", "fwd_median_ms", "fwd_std_ms",
              "e2e_median_ms", "e2e_std_ms", "fps"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(out_path).exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--models", default=None, help="comma names; default all in REGISTRY")
    ap.add_argument("--out", default=None, help="append rows to CSV")
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
        print(f"[measure] {cfg['name']} ({cfg['family']}, n_imgs={len(paths)}, "
              f"warmup={args.warmup}, reps={args.reps})", flush=True)
        sess = build_session(cfg["onnx"], threads=args.threads)
        rows.append(_measure(sess, cfg, inputs, args.warmup, args.reps))
        print(f"  -> fwd={rows[-1]['fwd_median_ms']:.1f}ms  "
              f"e2e={rows[-1]['e2e_median_ms']:.1f}ms  fps={rows[-1]['fps']:.1f}", flush=True)

    _print_table(rows)
    if args.out:
        _append_csv(rows, args.out)
        print(f"[csv] appended {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
