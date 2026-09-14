"""S1-A 探针续（2026-09-10）：QDQ 注册表缺 Sigmoid/Mul → 换 QOperator 能否让量化域连续？

上一轮的结论（scripts/probe_qdq_fusion.py + kernel_profile.csv）：
    naive static 的 YOLOv8n 有 64 个 Conv，只融合出 7 个 QLinearConv，另 57 个退回 FP32，
    还额外背了 636 个转换（Q/DQ）算子。
根因（本轮从 ORT 源码查实，不再是相关性的猜测）：
    PyTorch 把 SiLU 导成 **Sigmoid + Mul**；而 ORT 的 **QDQ 注册表里没有**
    Sigmoid / Mul / Add / Concat（注册表里却有它们在 QOperator 下的实现）。
    Conv 的输出喂给一个"没有 QDQ 量化器"的算子时，ORT 没地方放 QuantizeLinear，
    于是 `DQ -> Conv -> Q` 这个融合模式凑不齐，Conv 只能留在 FP32。
    融合成功的那 7 个，输出喂的是 **Reshape**——而 Reshape 恰好在 QDQ 注册表里。
    一个注册表成员关系就解释了 7/64 的全部差异。

本探针检验该解释给出的**可证伪推论**：
    若换成 QuantFormat.QOperator（QLinearConv 支持融合激活，QLinearSigmoid /
    QLinearMul 都在注册表里），量化域应能贯穿全网 → 整数 Conv 数应接近 64/64。

同时给出四档 forward-only 延迟，四者同 runtime（ORT CPU EP）、同硬件、同会话配置
（16 intra-op 线程、20 次预热、3 轮 × 8 图 × 30 次、取"轮内中位数再取轮间中位数"）：
    fp32 / static-QDQ / static-QOperator / dynamic
于是"INT8 在 CPU 上到底能不能更快"这个问题，就在同一个 runtime 内被回答，
不需要混用厂商数字，也不需要跨 runtime 比。

用法（一次性出数脚本，无单测；幂等性由"重跑得到同一表格"保证）：
    python scripts/probe_qdq_operator.py
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

from eval_common import IMGSZ, build_session, list_images, measure_forward, preprocess  # noqa: E402
from probe_qdq_fusion import DEFAULT_THREADS, optimized_hist  # noqa: E402
from protocol_io import image_paths_from_manifest, load_int8_triplet  # noqa: E402
from quantize_int8 import _CalibReader, _exclude_list, model_input_name  # noqa: E402

CALIB = ROOT / "data" / "coco128" / "coco128" / "images" / "train2017"
MODEL = "yolov8n"          # 可用 --model 换（第二步做 YOLOv8s 复现）
FP32 = ROOT / "data" / f"{MODEL}.onnx"
QDQ = ROOT / "data" / f"{MODEL}_static.onnx"
DYN = ROOT / "data" / f"{MODEL}_dynamic.onnx"
QOP = ROOT / "data" / f"{MODEL}_static_qop.onnx"
QOP_SEL = ROOT / "data" / f"{MODEL}_static_qop_sel.onnx"
QDQ_SEL = ROOT / "data" / f"{MODEL}_static_sel.onnx"   # 已有存档：QDQ + 保头
OUT_CSV = ROOT / "results" / f"probe_qdq_format_{MODEL}.csv"


def use_model(name: str) -> None:
    """把六个路径重绑到另一个模型（第二步的 YOLOv8s 复现靠它）。"""
    global MODEL, FP32, QDQ, DYN, QOP, QOP_SEL, QDQ_SEL, OUT_CSV
    MODEL = name
    FP32 = ROOT / "data" / f"{name}.onnx"
    QDQ = ROOT / "data" / f"{name}_static.onnx"
    DYN = ROOT / "data" / f"{name}_dynamic.onnx"
    QOP = ROOT / "data" / f"{name}_static_qop.onnx"
    QOP_SEL = ROOT / "data" / f"{name}_static_qop_sel.onnx"
    QDQ_SEL = ROOT / "data" / f"{name}_static_sel.onnx"
    OUT_CSV = ROOT / "results" / f"probe_qdq_format_{name}.csv"

# 关心的算子：整数核 vs FP32 核 vs 转换（Q/DQ）
INT_OPS = ["QLinearConv", "ConvInteger", "MatMulInteger", "QLinearMatMul",
           "QLinearSigmoid", "QLinearMul", "QLinearAdd", "QGemm"]
FP32_OPS = ["Conv", "MatMul", "Gemm", "Sigmoid", "Mul"]
CONV_OPS = ["QLinearConv", "ConvInteger", "Conv"]


def quantize_qoperator(src: Path, dst: Path, calib: Path, calib_n: int = 64,
                       exclude_head: bool = False) -> Path:
    """static 量化但用 QOperator 格式（QLinearConv 系列），权重/激活分开定类型。

    与 QDQ 的两处不同：
    1. quant_format=QuantFormat.QOperator —— 产出 QLinearConv 等融合算子，而非 Q/DQ 对。
    2. activation_type=QuantType.QUInt8 —— ONNX 的 QLinearConv 规定激活为 uint8
       （QDQ 下我们用 int8 激活是合法的，因为它只是 DQ 的 zero_point）。
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    name, shape = model_input_name(str(src))
    if shape and shape[0] not in (1, -1, None):
        raise ValueError(f"输入 batch 维不是 1：{name}{shape}")
    reader = _CalibReader(str(calib), name, calib_n)
    if not reader.paths:
        raise FileNotFoundError(f"{calib} 下没有校准图")
    quantize_static(
        str(src), str(dst), reader,
        quant_format=QuantFormat.QOperator,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=_exclude_list(str(src), True, exclude_head),
    )
    return dst


def census(path: Path, threads: int = DEFAULT_THREADS,
           optimized_path: str | None = None) -> dict:
    """转储满优化执行图，数整数核 / FP32 核 / 转换算子。

    只用 ORT_ENABLE_ALL 转储图，不建持久会话（optimized_hist 内部会释放）。
    """
    hist, _names = optimized_hist(str(path), threads=threads,
                                   optimized_path=optimized_path)
    row = {op: int(hist.get(op, 0)) for op in INT_OPS + FP32_OPS}
    row["conv_int"] = row["QLinearConv"] + row["ConvInteger"]
    row["conv_fp32"] = row["Conv"]
    row["conversion"] = int(hist.get("QuantizeLinear", 0) + hist.get("DequantizeLinear", 0))
    row["total_nodes"] = int(sum(hist.values()))
    return row


def timed_median(sess, inputs: list[np.ndarray], rounds: int, warmup: int, reps: int) -> tuple[float, list[float]]:
    """R 轮各自测一批，返回 (轮间中位数, 每轮中位数)。"""
    per_round = []
    for _ in range(rounds):
        ts = measure_forward(sess, inputs, warmup=warmup, reps=reps)
        per_round.append(st.median(ts))
    return st.median(per_round), per_round


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--only", default=None, help="只跑这些变体（逗号分隔），用于补测")
    ap.add_argument("--model", default="yolov8n", help="模型名（对应 data/<name>.onnx）")
    ap.add_argument("--imgs", default=str(CALIB), help="输入图像目录")
    ap.add_argument("--manifest", default=None,
                    help="按记录文件读取图像；传入后不走默认排序取前 n")
    ap.add_argument("--artifact-manifest", default=None,
                    help="显式 FP32/QDQ/QOperator 工件映射；启用后禁止自动导出")
    ap.add_argument("--optimized-dir", default=None,
                    help="保存每个工件的 ORT 优化执行图副本")
    ap.add_argument("--census-only", action="store_true",
                    help="只做节点普查，不重复做顺序延迟测量")
    ap.add_argument("--out", default=None,
                    help="输出 CSV；默认写入该模型的历史探针文件")
    args = ap.parse_args()
    use_model(args.model)

    if args.artifact_manifest:
        # S1 已经把路径和哈希冻结；显式映射模式绝不调用 quantize_qoperator。
        artifacts = load_int8_triplet(args.artifact_manifest, ROOT, args.model)
        variants = [("fp32", artifacts["FP32"]),
                    ("static_QDQ_sel", artifacts["QDQ"]),
                    ("static_QOp_sel", artifacts["QOperator"])]
        print(f"[artifacts] verified explicit map: {args.artifact_manifest}")
    else:
        # 保留旧的本地补测入口；S2 不使用该分支，避免产生替代工件。
        for dst, exclude_head in ((QOP, False), (QOP_SEL, True)):
            if dst.exists() and dst.stat().st_mtime >= FP32.stat().st_mtime:
                continue
            t0 = time.perf_counter()
            quantize_qoperator(FP32, dst, CALIB, exclude_head=exclude_head)
            print(f"[quant] {dst.name}（保头={exclude_head}）  "
                  f"{FP32.stat().st_size / 1e6:.1f}MB -> {dst.stat().st_size / 1e6:.1f}MB  "
                  f"({time.perf_counter() - t0:.0f}s)")
        variants = [("fp32", FP32), ("static_QDQ", QDQ), ("static_QDQ_sel", QDQ_SEL),
                    ("static_QOp", QOP), ("static_QOp_sel", QOP_SEL), ("dynamic", DYN)]
    if args.only:
        keep = {t.strip() for t in args.only.split(",")}
        variants = [v for v in variants if v[0] in keep]
        if not variants:
            raise SystemExit(f"--only {args.only} 没匹配到任何变体")
    if args.census_only:
        paths = []
        inputs = []
        print(f"[in] census-only；线程={args.threads}；不做延迟测量\n")
    else:
        paths = (image_paths_from_manifest(args.manifest, args.imgs, args.n_imgs)
                 if args.manifest else list_images(str(args.imgs), args.n_imgs))
        inputs = [preprocess(str(p), IMGSZ) for p in paths]
        print(f"[in] {len(inputs)} 张图（{Path(args.imgs).name}），线程={args.threads}，"
              f"预热={args.warmup}，{args.rounds} 轮 × {args.reps} 次\n")

    rows = []
    for tag, path in variants:
        print(f"===== {tag}  ({path.name}) =====", flush=True)
        optimized_path = None
        if args.optimized_dir:
            optimized_path = str(Path(args.optimized_dir) / f"{MODEL}_{tag}.onnx")
        c = census(path, threads=args.threads, optimized_path=optimized_path)
        print(f"  整数核: QLinearConv={c['QLinearConv']} ConvInteger={c['ConvInteger']} "
              f"QLinearSigmoid={c['QLinearSigmoid']} QLinearMul={c['QLinearMul']}")
        print(f"  FP32 核: Conv={c['Conv']} Sigmoid={c['Sigmoid']} Mul={c['Mul']}")
        print(f"  转换(Q/DQ)={c['conversion']}  总节点={c['total_nodes']}")

        if args.census_only:
            med, per_round = float("nan"), []
            print("  [census-only] 跳过延迟测量")
        else:
            sess = build_session(str(path), threads=args.threads)
            med, per_round = timed_median(sess, inputs, args.rounds, args.warmup, args.reps)
            print(f"  forward-only 中位延迟 = {med:.3f} ms   每轮中位={[round(x, 3) for x in per_round]}")
        rows.append({"model": MODEL, "variant": tag, "file": path.name,
                     "conv_int": c["conv_int"], "conv_fp32": c["conv_fp32"],
                     "conversion": c["conversion"], "total_nodes": c["total_nodes"],
                     "fwd_ms_median": round(med, 4),
                     "round_medians": "|".join(f"{x:.3f}" for x in per_round),
                     "threads": args.threads, "warmup": args.warmup,
                     "rounds": args.rounds, "reps": args.reps, "n_imgs": len(inputs)})
        print()

    if not args.census_only:
        base = rows[0]["fwd_ms_median"]
        print("===== 同 runtime（ORT CPU EP）内的相对速度 =====")
        for r in rows:
            print(f"  {r['variant']:12s} {r['fwd_ms_median']:8.3f} ms   "
                  f"相对 fp32 = {base / r['fwd_ms_median']:.3f}×   "
                  f"整数Conv={r['conv_int']:3d} / FP32 Conv={r['conv_fp32']:3d}")

    out_csv = Path(args.out) if args.out else OUT_CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    try:
        shown = out_csv.relative_to(ROOT)
    except ValueError:
        shown = out_csv
    print(f"\n[csv] -> {shown}")


if __name__ == "__main__":
    main()
