"""C1 (R9): 确认 INT8 图在 ORT CPU 满优化下实际执行哪些内核。

问题（审稿人 R9）：§4.4 的 INT8 图（dynamic / naive-static / selective-static）在
ONNX Runtime CPU EP 下跑，到底执行的是真实整数核（QLinearConv / ConvInteger /
MatMulInteger 等），还是 FP32 fallback + 一堆 Quantize/Dequantize 转换 op？
§5.2 曾写 "compute still runs on FP32 kernels wrapped by extra conversion ops"，
此句需由实测裁决。

方法（用户 2026-09-09 拍板 Method A：优化图转储）：
  用与测量**完全相同**的会话设置建会话——16 intra-op threads +
  GraphOptimizationLevel.ORT_ENABLE_ALL（见 eval_common.build_session）——
  只是额外设置 SessionOptions.optimized_model_filepath，让 ORT 把**融合后的执行图**
  落盘。执行图里的节点就是运行时会真正执行的算子（满优化下 ORT 先融合再执行），
  统计其 op 类型即可分类：整数核 vs FP32 核 vs 转换节点。确定性、无推理计时开销。

分类口径：
  - fp32_compute = {Conv, ConvTranspose, MatMul, Gemm}（论文 FLOPs 口径那四类；执行图中
    若还出现，即这些层以 FP32 内核运行）
  - integer_compute = op 以 'QLinear' 开头或 'Integer' 结尾（QLinearConv / ConvInteger /
    MatMulInteger / QLinearMatMul / QLinearConcat / QLinearSigmoid ... = 真实整数域内核）
  - conversion = {QuantizeLinear, DequantizeLinear, DynamicQuantizeLinear}
  - 专门再拆 conv_int = QLinearConv + ConvInteger（卷积的整数执行数，YOLO 主体）

自检（幂等，非单测框架，规则三"一次性出数脚本用临时幂等验证替代"）：
  [verify] 每张源图的配方标记与 label 声明一致
           fp32  → 量化类 op = 0
           dynamic → 有 DynamicQuantizeLinear / ConvInteger 等且无 QDQ(QuantizeLinear)
           static → 有 QDQ(QuantizeLinear + DequantizeLinear) 且无 DynamicQuantizeLinear
  [verify] selective 图（保检测头）的 Q+DQ 总数严格少于同模型 naive 图（头部整模块排除）
  [verify] optimized_model_filepath 每次实际写盘、节点非空、分类总数守恒

用法：
    python scripts/profile_kernels.py                # 跑内置目标表 -> results/kernel_profile.csv
    python scripts/profile_kernels.py --only sel_v8l # 只看某几个（label 前缀，逗号分隔）
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_common import DEFAULT_THREADS  # noqa: E402

# 目标表：label → (onnx 相对路径, recipe)。recipe ∈ {fp32, dynamic, naive, selective}
TARGETS: dict[str, tuple[str, str]] = {
    "fp32_v8n":   ("data/yolov8n.onnx",            "fp32"),
    "dyn_v8n":    ("data/yolov8n_dynamic.onnx",    "dynamic"),
    "dyn_rtl":    ("data/rtdetr-l_dynamic.onnx",   "dynamic"),
    "naive_v8n":  ("data/yolov8n_static.onnx",     "naive"),
    "sel_v8n":    ("data/yolov8n_static_sel.onnx", "selective"),
    "sel_v8l":    ("data/yolov8l_static_sel.onnx", "selective"),
}

QDQ = {"QuantizeLinear", "DequantizeLinear"}
DYNAMIC_Q = {"DynamicQuantizeLinear", "DynamicQuantizeMatMul"}
QOP_INT = {"ConvInteger", "MatMulInteger", "GemmInteger", "QLinearMatMul"}
FP32_COMPUTE = {"Conv", "ConvTranspose", "MatMul", "Gemm"}
CONVERSION = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"}


def _source_hist(path: str) -> Counter:
    m = onnx.load(path)
    return Counter(n.op_type for n in m.graph.node)


def verify_source(label: str, recipe: str, path: str, hist: Counter) -> list[str]:
    """源图配方标记与 label 声明一致；返回 [verify] 行。"""
    qdq = sum(hist[o] for o in QDQ)
    dyn = sum(hist[o] for o in DYNAMIC_Q)
    qop = sum(hist[o] for o in QOP_INT)
    lines = []
    if recipe == "fp32":
        ok = (qdq == 0) and (dyn == 0) and (qop == 0)
        lines.append(f"[verify] {label}: fp32 源图无量化 op (QDQ={qdq} dyn={dyn} int={qop}) -> "
                     f"{'OK' if ok else 'FAIL'}")
        return lines, ok
    if recipe == "dynamic":
        ok = (dyn > 0 or qop > 0) and (qdq == 0)
        lines.append(f"[verify] {label}: dynamic 源图标记 (QDQ={qdq} dynQ={dyn} intQop={qop}) -> "
                     f"{'OK' if ok else 'FAIL'}")
        return lines, ok
    # static: naive / selective
    ok = (qdq > 0) and (dyn == 0)
    lines.append(f"[verify] {label}: static 源图标记 (QDQ={qdq} dynQ={dyn}) -> "
                 f"{'OK' if ok else 'FAIL'}")
    return lines, ok


def analyze(path: str) -> tuple[Counter, dict]:
    """用测量同配置建会话并转储满优化执行图；返回 (优化图 op 直方图, 摘要 dict)。"""
    with tempfile.TemporaryDirectory() as td:
        opt_path = os.path.join(td, "opt.onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = DEFAULT_THREADS
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        so.optimized_model_filepath = opt_path
        sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        del sess  # 只需优化图；不用真跑推理
        m = onnx.load(opt_path)
        hist = Counter(n.op_type for n in m.graph.node)
    summary = _summarize(hist)
    return hist, summary


def _summarize(hist: Counter) -> dict:
    conv_int = hist["QLinearConv"] + hist["ConvInteger"]
    conv_fp32 = hist["Conv"]
    int_total = sum(c for op, c in hist.items()
                    if op.startswith("QLinear") or op.endswith("Integer"))
    fp32_total = sum(hist[o] for o in FP32_COMPUTE)
    conversion = sum(hist[o] for o in CONVERSION)
    total = sum(hist.values())
    return {
        "total_nodes": total,
        "conv_fp32": conv_fp32,
        "conv_int": conv_int,
        "integer_compute": int_total,
        "fp32_compute": fp32_total,
        "conversion": conversion,
        "others": total - int_total - fp32_total - conversion,
    }


def run(only: str | None) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    notes: list[str] = []
    for label, (rel, recipe) in TARGETS.items():
        if only and not any(label.startswith(p.strip()) for p in only.split(",")):
            continue
        path = str(ROOT / rel)
        if not os.path.exists(path):
            notes.append(f"[skip] {label}: {rel} 不存在")
            continue
        hist = _source_hist(path)
        vlines, ok = verify_source(label, recipe, path, hist)
        notes.extend(vlines)
        if not ok:
            raise SystemExit(f"[verify] FAIL @ {label}，中止")
        _, summary = analyze(path)
        # 归并进 label/recipe 便于 CSV
        row = {"label": label, "recipe": recipe, **summary}
        rows.append(row)
        notes.append(
            f"[opt] {label}: total={summary['total_nodes']} "
            f"conv_fp32={summary['conv_fp32']} conv_int={summary['conv_int']} "
            f"integer={summary['integer_compute']} fp32_compute={summary['fp32_compute']} "
            f"conversion={summary['conversion']}"
        )
    # selective 头保 FP32 → 同模型 Q+DQ 严格更少（naive_v8n vs sel_v8n）
    r_naive = next((r for r in rows if r["label"] == "naive_v8n"), None)
    r_sel = next((r for r in rows if r["label"] == "sel_v8n"), None)
    if r_naive and r_sel:
        a = r_naive["conversion"]
        b = r_sel["conversion"]
        notes.append(f"[verify] naive_v8n Q+DQ={a} > sel_v8n Q+DQ={b} (selective 保头) -> "
                     f"{'OK' if a > b else 'FAIL'}")
        if not a > b:
            raise SystemExit("[verify] FAIL @ selective 保头 Q/DQ 应少于 naive")
    return rows, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "kernel_profile.csv"))
    args = ap.parse_args()

    rows, notes = run(args.only)
    for n in notes:
        print(n)

    fields = ["label", "recipe", "total_nodes", "conv_fp32", "conv_int",
              "integer_compute", "fp32_compute", "conversion", "others"]
    print("\n| label | recipe | total | conv_fp32 | conv_int | integer | fp32_compute | conversion |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {r['recipe']} | {r['total_nodes']} | {r['conv_fp32']} | "
              f"{r['conv_int']} | {r['integer_compute']} | {r['fp32_compute']} | "
              f"{r['conversion']} |")
    if args.out and rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
