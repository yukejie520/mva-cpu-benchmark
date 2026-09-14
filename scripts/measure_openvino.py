"""跨运行时 2×2（2026-09-10）：同一份 ONNX，换 Runtime（ORT / OpenVINO）不换模型。

要回答的问题：
    上一轮已证：在我们自己的 QDQ 导出路径下 INT8 比 FP32 慢（0.42×），换成 QOperator
    格式则快（1.61×）。那么"慢"到底是不是 ORT CPU EP 自己的内核选择问题？
    换一个 VNNI-aware 的运行时（OpenVINO），拿**同一份 ONNX 文件**再测一遍即可分离：
      - 同 runtime 内比 INT8 vs FP32  → 问"这个运行时能不能吃到 INT8 的收益"
      - 同精度跨 runtime 比           → 问"运行时本身差多少"
    **绝不跨 runtime 比 INT8 vs FP32**（那是混了运行时与精度两个因子，正是我们批评别人的做法）。

协议（与上一轮完全一致，且全部放进同一次交错轮转）：
    8 张图 × 30 次 × 5 轮；每轮轮转变体顺序；16 线程；20 次预热；
    统计量 = 轮内配对比值（同轮内做除法，消掉该轮的公共热状态）；每轮前后记有效频率。
    用 forward-only；端到端（含 NMS 后处理）另有固定开销，不混进这张表。

用法：
    python scripts/measure_openvino.py --rounds 5
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_common import IMGSZ, build_session, list_images, preprocess  # noqa: E402
from measure_interleaved import cpu_mhz, median_of_medians, rotation_orders, summarize  # noqa: E402

CALIB = ROOT / "data" / "coco128" / "coco128" / "images" / "train2017"
D = ROOT / "data"

# 注意 QOp（QOperator）那一列只在 ORT 下有：ORT 导出的是 com.microsoft 私有域的
# QLinearAdd/QLinearMul/QLinearSigmoid，且张量元素类型留成 dynamic，OpenVINO 的 ONNX
# 前端直接拒收（要求具体 i8/u8）。所以"最快的那条 INT8 路径"是 ORT 私有格式、不可移植——
# 这本身就是要写进正文的一条结论，不是实验失败。
VARIANTS = [
    ("ORT_FP32", "ort", D / "yolov8n.onnx"),
    ("ORT_QDQ", "ort", D / "yolov8n_static.onnx"),
    ("ORT_QDQ_sel", "ort", D / "yolov8n_static_sel.onnx"),
    ("ORT_QOp", "ort", D / "yolov8n_static_qop.onnx"),
    ("ORT_QOp_sel", "ort", D / "yolov8n_static_qop_sel.onnx"),
    ("OV_FP32", "ov", D / "yolov8n.onnx"),
    ("OV_QDQ", "ov", D / "yolov8n_static.onnx"),
    ("OV_QDQ_sel", "ov", D / "yolov8n_static_sel.onnx"),
]


class Runner:
    """把 ORT 与 OpenVINO 的前向调用统一成 runner.run(x) -> ndarray。

    为什么要包一层：本实验的全部意义在于"除了运行时，别的都一样"。若两条分支各自写一套
    计时循环，预热/计时/取输出的细节就会漂移，测出来的差异就说不清是运行时的还是计时代码的。
    """

    def __init__(self, kind: str, path: Path, threads: int = 16):
        self.kind = kind
        if kind == "ort":
            self.sess = build_session(str(path), threads=threads)
            self.inp = self.sess.get_inputs()[0].name
            self.outs = [o.name for o in self.sess.get_outputs()]
        elif kind == "ov":
            import openvino as ov

            core = ov.Core()
            core.set_property("CPU", {"INFERENCE_NUM_THREADS": threads,
                                      "PERFORMANCE_HINT": "LATENCY",
                                      "NUM_STREAMS": 1})
            self.compiled = core.compile_model(str(path), "CPU")
            self.req = self.compiled.create_infer_request()
            self.out_key = self.compiled.output(0)
        else:
            raise ValueError(kind)

    def run(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "ort":
            return self.sess.run(self.outs, {self.inp: x})[0]
        res = self.req.infer({0: x})
        return np.asarray(res[self.out_key])


def timed_median(runner: Runner, inputs: list[np.ndarray], warmup: int, reps: int) -> float:
    """与 eval_common.measure_forward 同口径：先预热，再逐图重复计时，取中位数。"""
    for _ in range(warmup):
        runner.run(inputs[0])
    ts = []
    for x in inputs:
        for _ in range(reps):
            t0 = time.perf_counter()
            runner.run(x)
            ts.append((time.perf_counter() - t0) * 1e3)
    return st.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "results" / "openvino_2x2.csv"))
    args = ap.parse_args()

    inputs = [preprocess(str(p), IMGSZ) for p in list_images(str(CALIB), args.n_imgs)]
    labels = [v[0] for v in VARIANTS]
    runners = {tag: Runner(kind, path, args.threads) for tag, kind, path in VARIANTS}

    # 接线自检：同一份 FP32 ONNX 在两个运行时下的原始输出应当几乎相同。
    # 若这里差异大，说明输入布局/输出取值接错了，后面所有跨运行时比较都不成立。
    a = runners["ORT_FP32"].run(inputs[0])
    b = runners["OV_FP32"].run(inputs[0])
    print(f"[check] ORT_FP32 输出{a.shape} vs OV_FP32 输出{b.shape}  "
          f"max|Δ|={np.abs(a - b).max():.3e}  mean|Δ|={np.abs(a - b).mean():.3e}")
    print(f"[freq] 起始有效频率 {cpu_mhz():.0f} MHz（标称基频 2200）\n")

    raw, per_round = [], {t: [] for t in labels}
    for r, order in enumerate(rotation_orders(len(VARIANTS), args.rounds)):
        print(f"---- 第 {r + 1}/{args.rounds} 轮  顺序: {[labels[i] for i in order]} ----",
              flush=True)
        for i in order:
            tag = labels[i]
            before = cpu_mhz()
            med = timed_median(runners[tag], inputs, args.warmup, args.reps)
            after = cpu_mhz()
            per_round[tag].append(med)
            raw.append({"round": r + 1, "position": order.index(i), "variant": tag,
                        "median_ms": round(med, 4), "n_samples": len(inputs) * args.reps,
                        "freq_before_mhz": round(before, 1), "freq_after_mhz": round(after, 1)})
            print(f"   {tag:14s} {med:9.3f} ms   freq {before:.0f}->{after:.0f} MHz", flush=True)
        print()

    # 每条对比只允许变一个因子：要么同 runtime 换精度，要么同精度换 runtime。
    # 跨 runtime 又跨精度地比（例如 OV_INT8 vs ORT_FP32）是被禁止的——那正是我们批评别人的做法。
    comparisons = [
        ("精度: OV_QDQ / OV_FP32", "OV_FP32", "OV_QDQ", "同 runtime(OV) 换精度 ★核心格"),
        ("精度: ORT_QDQ / ORT_FP32", "ORT_FP32", "ORT_QDQ", "同 runtime(ORT) 换精度"),
        ("精度: OV_QDQ_sel / OV_FP32", "OV_FP32", "OV_QDQ_sel", "同 runtime(OV) 保头换精度"),
        ("精度: ORT_QDQ_sel / ORT_FP32", "ORT_FP32", "ORT_QDQ_sel", "同 runtime(ORT) 保头换精度"),
        ("精度: ORT_QOp / ORT_FP32", "ORT_FP32", "ORT_QOp", "ORT 私有格式的最快路径"),
        ("运行时: OV_FP32 / ORT_FP32", "ORT_FP32", "OV_FP32", "同精度(FP32) 换运行时"),
        ("运行时: OV_QDQ / ORT_QDQ", "ORT_QDQ", "OV_QDQ", "同精度同格式换运行时"),
        ("运行时: OV_QDQ_sel / ORT_QDQ_sel", "ORT_QDQ_sel", "OV_QDQ_sel", "同精度同格式换运行时(保头)"),
    ]
    print("===== 轮内配对比值（>1 表示分子更快）=====")
    print(f"  {'对比':34s} {'比值中位':>9s} {'最小':>7s} {'最大':>7s} {'每轮>1':>7s}  说明")
    summary = []
    for name, base_tag, var_tag, note in comparisons:
        s = summarize(per_round[base_tag], per_round[var_tag])
        s.update({"comparison": name, "base": base_tag, "var": var_tag, "note": note,
                  "rounds": args.rounds})
        summary.append(s)
        print(f"  {name:34s} {s['ratio_median']:9.3f} {s['ratio_min']:7.3f} "
              f"{s['ratio_max']:7.3f} {'是' if s['ratio_all_gt1'] else '否':>7s}  {note}")

    print("\n  （绝对值，中位延迟 ms）")
    for tag in labels:
        print(f"    {tag:14s} {median_of_medians(per_round[tag]):9.3f}   "
              f"轮内: {'|'.join(f'{x:.2f}' for x in per_round[tag])}")

    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)
    sout = out.with_name(out.stem + "_summary.csv")
    with open(sout, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\n[csv] -> {out.name}, {sout.name}")


if __name__ == "__main__":
    main()
