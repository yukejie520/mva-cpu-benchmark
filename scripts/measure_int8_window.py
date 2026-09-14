"""INT8 同窗口稳健延迟：FP32 与 dynamic/static INT8 在同一次运行里逐一测多轮。

为什么同窗口：账本 §3 要报「量化后延迟收益」。消费级 CPU 睿频/温度会让跨时段
单窗口延迟漂移 ~30%（见 notes/04），FP32 的正式值是之前窗口的。要公平比 Δ，
必须在**同一次运行**里把 FP32 与 12 个 INT8 一起按相同多轮协议测，让时钟漂移
对两侧影响一致，Δ 才是量化本身的效果。

复用：measure_robust 的多轮聚合(round_stats) + measure 的逐模型计测(_measure)，
session 指向被测文件(FP32 或 INT8)，cfg 的 onnx 保留 FP32 源 → params_M/gflops
取 FP32（避免 INT8 QDQ 图里 Quantize/Dequantize 算子把 FLOPs 算虚高）。
kind/family 沿用 FP32 源，YOLO INT8 同样含 Python NMS、RT-DETR INT8 端到端。

用法：
    python scripts/measure_int8_window.py --rounds 3 \
        --imgs data/coco128/coco128/images/train2017 --out results/int8_latency_window.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from eval_common import IMGSZ, build_session, list_images, preprocess  # noqa: E402
from measure import REGISTRY, _measure  # noqa: E402
from measure_robust import CSV_FIELDS, _append_csv, round_stats  # noqa: E402

# static 一律测 selective 配方（_static_sel = 排输出Concat+检测头，骨干/颈部 INT8，
# 账本 §3 发现①延伸/双配方拍板）：naive(_static) 与其延迟几乎相同（检测头算力占比 <1%），
# 只取 canonical 一个延迟，避免表里塞两个几乎一样的列。RT-DETR 无 _static_sel →
# main 里 Path.exists() 检查自动跳过 → static 行自然 N/A，与账本发现②一致。
INT8_SUFFIXES = [("dynamic", "_dynamic.onnx"), ("static", "_static_sel.onnx")]

# dynamic INT8 在 CPU 上对小模型已慢 ~8-10×（2026-09-06 实测），大模型会到数秒/次；
# 完整 3 轮协议成本失控。方案 A：只给「小模型」测 dynamic，大模型 dynamic 只留体积/结论。
BIG_DYNAMIC_BASES = {"YOLOv8m", "YOLOv8l", "RT-DETR-l", "RT-DETR-x"}

# RT-DETR 的 static（QDQ）数值崩溃：骨干/编码器区 scale=0/NaN，输出全常数；已试 per-tensor、
# 排除 decoder+框头均无效（账本 §3 发现②，2026-09-06）。用户拍板不深挖、报 N/A →
# 从测速表**永远剔除**，不测一个数值崩溃模型的"延迟"（无意义且误导）。
COLLAPSED_STATIC = {"RT-DETR-l-static", "RT-DETR-x-static"}


def targets() -> list[dict]:
    """拼出同窗口被测表：6 FP32 + 12 INT8，cfg.onnx=FP32 源（供参数/FLOPs）。"""
    out = []
    for c in REGISTRY:
        stem = Path(c["onnx"]).stem  # e.g. yolov8n
        out.append({"name": f"{c['name']}-FP32", "family": c["family"], "kind": c["kind"],
                    "onnx": c["onnx"], "run_onnx": c["onnx"]})
        for label, suffix in INT8_SUFFIXES:
            p = Path(c["onnx"]).parent / f"{stem}{suffix}"
            out.append({"name": f"{c['name']}-{label}", "family": c["family"], "kind": c["kind"],
                        "onnx": c["onnx"], "run_onnx": str(p)})
    return out


def select_targets(ts: list[dict], skip_big_dynamic: bool, only: str | None = None) -> list[dict]:
    """按方案 A 过滤：崩溃 static 永远剔除；skip_big_dynamic 剔除大模型 dynamic。

    only（如 "RT-DETR-l-dynamic"）用于单独降协议测某几个目标：一旦指定，
    其它过滤（除崩溃剔除）不再生效，只返回点名者——否则 --only 会被
    skip_big_dynamic 抢先砍掉 RT-DETR 的 dynamic。
    """
    out = []
    for t in ts:
        if t["name"] in COLLAPSED_STATIC:
            continue
        if only:
            if any(t["name"].startswith(p.strip()) for p in only.split(",")):
                out.append(t)
            continue
        base = t["name"].rsplit("-", 1)[0]
        if skip_big_dynamic and t["name"].endswith("-dynamic") and base in BIG_DYNAMIC_BASES:
            continue  # 只剔除「大模型 dynamic」这一档
        out.append(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--skip-big-dynamic", action="store_true",
                    help="方案 A：只测 n/s 的 dynamic，跳过 m/l/RT-DETR 的 dynamic")
    ap.add_argument("--only", default=None,
                    help="只测这些 name 前缀（逗号分隔）。用于给 RT-DETR dynamic 单独降协议/降轮次")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = list_images(args.imgs, args.n_imgs)
    if not paths:
        raise FileNotFoundError(f"no jpg under {args.imgs}")
    inputs = [preprocess(str(p), IMGSZ) for p in paths]

    rows = []
    for t in select_targets(targets(), args.skip_big_dynamic, args.only):
        if not Path(t["run_onnx"]).exists():
            print(f"[skip] {t['name']}: {t['run_onnx']} 不存在", flush=True)
            continue
        print(f"[int8-window] {t['name']} rounds={args.rounds}", flush=True)
        sess = build_session(t["run_onnx"])  # FP32 或 INT8 都由同一入口建会话
        cfg = {"name": t["name"], "family": t["family"], "kind": t["kind"], "onnx": t["onnx"]}
        fwd_meds, e2e_meds = [], []
        for r in range(args.rounds):
            row = _measure(sess, cfg, inputs, args.warmup, args.reps)
            fwd_meds.append(row["fwd_median_ms"])
            e2e_meds.append(row["e2e_median_ms"])
            print(f"  round {r + 1}: fwd={row['fwd_median_ms']:.1f} "
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
        print(f"  -> e2e 中位={em:.1f}ms (跨轮 {elo:.1f}~{ehi:.1f})", flush=True)

    print("| name | e2e median_ms | range | fps |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['e2e_median_ms']:.1f} | "
              f"{r['e2e_lo_ms']:.1f}~{r['e2e_hi_ms']:.1f} | {r['fps']:.1f} |")
    if args.out:
        _append_csv(rows, args.out)
        print(f"[csv] appended {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
