"""小模型 FP32 交错切片（2026-09-11）：给 Pareto 边的**延迟腿**配一个配对 CI。

为什么需要它：
    §4.3 里 YOLO11n -> YOLOv8n 那条支配边，精度腿按 §3.4 的规则判为"无正向证据"
    （配对 CI 下界为负），于是**延迟腿成了唯一承重腿**。而 Table 2 里这条腿很软：
    YOLO11n 29.568 ms [29.510, 30.237]，YOLOv8n 31.459 ms [29.346, 31.635]，
    区间大幅重叠，差 6%。这两个数是**跨轮**口径（三轮各自取中位再取中位），
    没有轮内配对，因此无法给出"同一热状态下 v8n 比 YOLO11n 慢多少"的区间。

怎么解决：轮内配对，和 measure_interleaved.py 同一套机制。
    每一轮里三个模型各跑一遍，轮内顺序逐轮旋转（round-robin），于是三者都经历
    相近的热状态。关键统计量是**轮内配对比值** r_i = t_v8n(第 i 轮) / t_11n(第 i 轮)，
    在同一轮内做除法，天然消掉该轮的公共频率/温度水平。
    报告 R 个比值的中位数、范围，以及对**轮**做有放回重采样得到的 95% CI。

    为什么不直接对样本做 bootstrap：同轮内的几百个 latency 样本共享同一个热状态，
    不是独立观测；把它们当独立会**低估**区间（正是 Table 2 区间重叠却仍被拿来
    下结论的根源）。独立的实验单元是**轮**，所以重采样单位也是轮。

口径与 measure.py 一致：端到端 = YOLO 前向 + Python NMS 解码（Pareto 轴用的就是这个）。
输入切片也沿用 coco128 前 8 张，与既有测量可比。

用法：
    python scripts/measure_interleaved_fp32.py --rounds 11 --n-imgs 8
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_common import IMGSZ, build_session, list_images, preprocess  # noqa: E402
from measure_interleaved import cpu_mhz, rotation_orders  # noqa: E402 复用同一套轮转与频率读数
from yolo_post import yolov8_nms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "data" / "coco128" / "coco128" / "images" / "train2017"
OUT_RAW = ROOT / "results" / "interleaved_fp32_small.csv"
OUT_SUM = ROOT / "results" / "interleaved_fp32_small_summary.csv"

# 顺序即轮转基准位置；三个都是 FP32（无量化后缀）。
MODELS = [("YOLO11n", ROOT / "data" / "yolo11n.onnx"),
          ("YOLOv8n", ROOT / "data" / "yolov8n.onnx"),
          ("YOLOv8s", ROOT / "data" / "yolov8s.onnx")]

# 需要给出配对 CI 的有序对（分子 / 分母）。第一对是 §4.3 那条支配边的延迟腿。
PAIRS = [("YOLOv8n", "YOLO11n"), ("YOLOv8s", "YOLO11n"), ("YOLOv8s", "YOLOv8n")]


def measure_e2e(sess, inputs: list[np.ndarray], warmup: int, reps: int) -> list[float]:
    """端到端延迟样本(ms)：前向 + NMS 解码，口径与 measure.py 一致。"""
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    for _ in range(warmup):
        yolov8_nms(sess.run(outs, {inp: inputs[0]})[0])
    times: list[float] = []
    for x in inputs:
        for _ in range(reps):
            t0 = time.perf_counter()
            yolov8_nms(sess.run(outs, {inp: x})[0])
            times.append((time.perf_counter() - t0) * 1e3)
    return times


def boot_median_ci(ratios: list[float], n_boot: int = 20000, seed: int = 0,
                   alpha: float = 0.05) -> tuple[float, float]:
    """对**轮**做有放回重采样，得到配对比值中位数的 95% CI。

    纯函数，便于测试。ratios 的每个元素是"一轮的配对比值"，即一个独立的实验单元。
    """
    r = np.asarray(ratios, dtype=np.float64)
    if r.size == 0:
        raise ValueError("ratios 不能为空")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r.size, size=(n_boot, r.size))
    meds = np.median(r[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inputs = [preprocess(str(p), IMGSZ) for p in list_images(str(CALIB), args.n_imgs)]
    labels = [m for m, _ in MODELS]
    sess = {name: build_session(str(path), threads=args.threads) for name, path in MODELS}

    print(f"[in] {len(inputs)} 张图 {CALIB.name}；线程={args.threads}；预热={args.warmup}；"
          f"{args.rounds} 轮 × {args.reps} 次 × {len(inputs)} 图；口径=端到端(前向+NMS)")
    print(f"[freq] 起始有效频率 {cpu_mhz():.0f} MHz\n")

    raw_rows, per_round, freq = [], {m: [] for m in labels}, []
    for r, order in enumerate(rotation_orders(len(MODELS), args.rounds)):
        print(f"---- 第 {r + 1}/{args.rounds} 轮  轮内顺序: {[labels[i] for i in order]} ----",
              flush=True)
        for i in order:
            name, _ = MODELS[i]
            before = cpu_mhz()
            ts = measure_e2e(sess[name], inputs, args.warmup, args.reps)
            after = cpu_mhz()
            med = st.median(ts)
            per_round[name].append(med)
            freq.append((before + after) / 2)
            raw_rows.append({"round": r + 1, "order": order.index(i), "model": name,
                             "median_e2e_ms": round(med, 4), "n_samples": len(ts),
                             "freq_before_mhz": round(before, 1),
                             "freq_after_mhz": round(after, 1)})
            print(f"   {name:9s} {med:8.3f} ms   freq {before:.0f}->{after:.0f} MHz", flush=True)
        print()

    print("===== 轮内配对比值（同一轮内做除法）=====")
    summary = []
    for num, den in PAIRS:
        ratios = [a / b for a, b in zip(per_round[num], per_round[den])]
        lo, hi = boot_median_ci(ratios, n_boot=args.boot, seed=args.seed)
        med = st.median(ratios)
        summary.append({"numerator": num, "denominator": den,
                        "median_ms_num": round(st.median(per_round[num]), 3),
                        "median_ms_den": round(st.median(per_round[den]), 3),
                        "ratio_median": round(med, 4), "ratio_min": round(min(ratios), 4),
                        "ratio_max": round(max(ratios), 4),
                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "ci_excludes_1": bool(lo > 1.0 or hi < 1.0),
                        "rounds": args.rounds, "n_boot": args.boot, "seed": args.seed})
        s = summary[-1]
        print(f"  {num:9s} / {den:9s} 比值中位 {s['ratio_median']:.3f}  "
              f"范围 [{s['ratio_min']:.3f}, {s['ratio_max']:.3f}]  "
              f"95% CI [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]  "
              f"{'CI 不含 1（可信）' if s['ci_excludes_1'] else 'CI 含 1（未分离）'}")

    valid = [f for f in freq if f == f]
    if valid:
        print(f"\n[freq] 全程有效频率 中位={st.median(valid):.0f} MHz  "
              f"最低={min(valid):.0f}  最高={max(valid):.0f}")

    with open(OUT_RAW, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)
    with open(OUT_SUM, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\n[csv] -> {OUT_RAW.name}, {OUT_SUM.name}")


if __name__ == "__main__":
    main()
