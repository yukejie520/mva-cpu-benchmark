"""OpenVINO 线程配置探针（2026-09-10，一次性出数脚本）。

问题：上一轮八格表里所有 ORT 变体的轮内中位数都很稳（FP32 29.89/29.90/29.91/30.38/27.37），
而 OV_FP32 是 41.7 / 35.5 / 42.2 / 34.5 / 66.8 —— 五轮差了近一倍。拿一个自身不稳的分母去算
"OV_INT8 / OV_FP32"，结论就不可信。

已查到的线索：OV 在这台机器上 `ENABLE_CPU_PINNING = False`（默认不绑定线程）。
本机是 i7-14650HX，24 个逻辑核由 P 核（8 核 16 线程）与 E 核（8 核）混合组成。
线程若不绑定，会在两类核之间迁移，快照延迟就会大幅波动。

本探针比四种配置，每轮轮转、共 N 轮，看哪种配置的**轮内中位数极差**最小：
    A 现状    ：16 线程 + LATENCY + 1 stream（未绑定）
    B 绑定    ：A + ENABLE_CPU_PINNING=True
    C 只绑P核 ：B + SCHEDULING_CORE_TYPE=PCORE_ONLY
    D 吞吐    ：16 线程 + THROUGHPUT

判据：选极差最小者作为后续 OV 测量的正式配置；若都不稳，则如实说明该平台的 OV 延迟
不可用于比值结论，只报方向。

用法：
    python scripts/probe_ov_threads.py --rounds 5
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_common import IMGSZ, list_images, preprocess  # noqa: E402
from measure_interleaved import cpu_mhz  # noqa: E402

CALIB = ROOT / "data" / "coco128" / "coco128" / "images" / "train2017"
D = ROOT / "data"

MODELS = {"FP32": D / "yolov8n.onnx", "QDQ_INT8": D / "yolov8n_static.onnx"}

CONFIGS = {
    "A_现状": {"INFERENCE_NUM_THREADS": 16, "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1},
    "B_绑定": {"INFERENCE_NUM_THREADS": 16, "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
               "ENABLE_CPU_PINNING": True},
    "C_只绑P核": {"INFERENCE_NUM_THREADS": 16, "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
                  "ENABLE_CPU_PINNING": True, "SCHEDULING_CORE_TYPE": "PCORE_ONLY"},
    "D_吞吐": {"INFERENCE_NUM_THREADS": 16, "PERFORMANCE_HINT": "THROUGHPUT"},
}


def build(cfg: dict, path: Path):
    import openvino as ov

    core = ov.Core()
    core.set_property("CPU", cfg)
    compiled = core.compile_model(str(path), "CPU")
    req = compiled.create_infer_request()
    key = compiled.output(0)

    def run(x):
        return req.infer({0: x})[key]

    return run


def timed_median(run, inputs, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        run(inputs[0])
    return st.median([_time(run, x) for x in inputs for _ in range(reps)])


def _time(run, x) -> float:
    import time

    t0 = time.perf_counter()
    run(x)
    return (time.perf_counter() - t0) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n-imgs", type=int, default=8)
    args = ap.parse_args()

    inputs = [preprocess(str(p), IMGSZ) for p in list_images(str(CALIB), args.n_imgs)]
    combos = [(cname, mname) for cname in CONFIGS for mname in MODELS]
    runners = {(c, m): build(CONFIGS[c], MODELS[m]) for c, m in combos}

    print(f"{len(inputs)} 图 × {args.reps} 次 × {args.rounds} 轮；轮内轮转；"
          f"起始频率 {cpu_mhz():.0f} MHz\n")
    per_round = {k: [] for k in combos}
    for r in range(args.rounds):
        order = combos[r % len(combos):] + combos[:r % len(combos)]
        for key in order:
            per_round[key].append(timed_median(runners[key], inputs, args.warmup, args.reps))
        print(f"  第 {r+1} 轮完成  频率 {cpu_mhz():.0f} MHz", flush=True)

    print(f"\n{'配置 / 模型':22s} {'中位ms':>9s} {'最小':>9s} {'最大':>9s} {'极差%':>7s}"
          f"   轮内中位数")
    for key in combos:
        v = per_round[key]
        med, lo, hi = st.median(v), min(v), max(v)
        print(f"  {key[0] + ' / ' + key[1]:20s} {med:9.3f} {lo:9.3f} {hi:9.3f} "
              f"{(hi - lo) / med * 100:6.1f}%   {'|'.join(f'{x:.1f}' for x in v)}")

    print("\n判据：极差% 最小者作为后续 OV 测量的正式配置。")


if __name__ == "__main__":
    main()
