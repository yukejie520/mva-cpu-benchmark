"""从 ONNX 模型统计 参数量 / FLOPs（跨 YOLO/RT-DETR 两族统一口径）。

为什么不用 torch/thop：① 本机是纯 ONNX/ORT 链路，不该为统计再依赖 torch；
② 官方 mmcv 系上不了 py3.14；③ 从同一个 .onnx 数，YOLO 和 RT-DETR 口径天然一致。

口径说明：
- params：图内所有 initializer（权重/偏置/归一化系数）的元素总数 /1e6 → M。
  常量（shape/reshape 整数）也在内，但每常量仅几元素，对 M 级影响 <0.1%，可忽略。
- gflops：对 Conv/ConvTranspose/MatMul/Gemm 统计 2×MACs（×2 对齐 ultralytics 惯例）。
  形状来源两种，可互换：
  ① 静态（onnx_gflops）：shape_inference 后的中间张量形状；动态维被 Reshape 阻断时
     下游会漏算 → RT-DETR transformer 的线性层曾漏 102 个 Gemm（只有 92.54G 下界）。
  ② 运行时（onnx_gflops_runtime，推荐）：把静态拿不到形状的 FLOP 算子输入临时声明
     为图输出，用 onnxruntime 前向一次读出真实形状。能覆盖动态 Reshape 断链，给出
     640 输入下的真值（RT-DETR-l ≈ 110.2G，与 ultralytics 官方 model.info() 一致）。
- 输入固定为方形 imgsz（本实验 640）。形状仍拿不到的节点计入 skipped。

用法：
    python scripts/onnx_metrics.py data/rtdetr-l.onnx --runtime   # 推荐：运行时真值
    python scripts/onnx_metrics.py data/rtdetr-l.onnx             # 静态（保守，会漏）
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import onnx

# FLOP 算子集合（只统计这四个；其余 Reshape/Transpose/Softmax/归一化 <1% 忽略）
_FLOP_OPS = ("Conv", "ConvTranspose", "MatMul", "Gemm")


# ---------- 参数量 ----------

def onnx_params_m(onnx_path: str) -> float:
    """图内所有 initializer 的总元素数（百万）。"""
    m = onnx.load(onnx_path)
    total = 0
    for init in m.graph.initializer:
        total += int(np.prod(list(init.dims), dtype=np.int64)) if init.dims else 1
    return total / 1e6


# ---------- 形状相关 ----------

def _static_shape(dims, dyn_value: Optional[int] = 1) -> Optional[list[int]]:
    """取静态形状。dim_value>0 用之；dim_param（如导出时 batch 符号）视为 dyn_value。

    ultralytics 导出是 batch=1 静态图，出现 dim_param 的通常是 batch 符号，
    运行时恒为 1，所以默认按 1 处理可消掉绝大多数动态跳过。真·动态维返回 None。
    """
    if dims is None:
        return None
    out = []
    for d in dims.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            out.append(int(d.dim_value))
        elif d.HasField("dim_param") and dyn_value is not None:
            out.append(int(dyn_value))
        else:
            return None  # 真动态
    return out


def _attr(node, name: str, default):
    for a in node.attribute:
        if a.name == name:
            kind = a.type
            if kind == onnx.AttributeProto.INT:
                return int(a.i)
            if kind == onnx.AttributeProto.INTS:
                return list(a.ints)
            if kind == onnx.AttributeProto.FLOAT:
                return float(a.f)
    return default


def _shape_map(model) -> dict[str, list[int]]:
    """infer_shapes 后建立 name -> 静态形状。含动态维的条目丢弃。"""
    smap: dict[str, list[int]] = {}
    for vi in list(model.graph.value_info):
        s = _static_shape(vi.type.tensor_type.shape)
        if s:
            smap[vi.name] = s
    for inp in model.graph.input:
        s = _static_shape(inp.type.tensor_type.shape)
        if s:
            smap[inp.name] = s
    return smap


def _init_shapes(model) -> dict[str, list[int]]:
    """initializer（权重/常量）name -> dims。权重形状不依赖输入，恒为静态。"""
    return {init.name: [int(d) for d in init.dims] for init in model.graph.initializer}


# ---------- 逐算子 FLOPs（共用：喂一张 name->shape 表即可） ----------

def _flops_conv(node, shapes) -> Optional[float]:
    x = shapes.get(node.input[0])
    w = shapes.get(node.input[1])
    if x is None or w is None or len(x) != 4 or len(w) != 4:
        return None
    n, c_in = x[0], x[1]
    out_c, kh, kw = w[0], w[2], w[3]
    group = _attr(node, "group", 1)
    strides = _attr(node, "strides", [1, 1])
    pads = _attr(node, "pads", [0, 0, 0, 0])
    dil = _attr(node, "dilations", [1, 1])
    h, w = x[2], x[3]
    oh = (h + pads[0] + pads[2] - dil[0] * (kh - 1) - 1) // strides[0] + 1
    ow = (w + pads[1] + pads[3] - dil[1] * (kw - 1) - 1) // strides[1] + 1
    macs = n * out_c * oh * ow * (c_in // group) * kh * kw
    # ConvTranspose 与 Conv 的乘加量接近，用同式近似（不计分组反卷积特殊性）
    return 2.0 * macs if macs > 0 else None


def _flops_matmul(node, shapes) -> Optional[float]:
    """通用 MatMul：取末尾两维 M@K·K@N，前导 batch 维做广播合并。

    覆盖 (M,K)@(K,N)、(B,M,K)@(B,K,N)、RT-DETR attention 的 4D
    (B,H,S,D)@(B,H,D,S) 等。广播近似：前导维逐对取 max（标准广播语义）。
    """
    a = shapes.get(node.input[0])
    b = shapes.get(node.input[1])
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return None
    if a[-1] != b[-2]:
        return None
    k = a[-1]
    m_, n_ = a[-2], b[-1]

    def bcast_prod(d1, d2) -> int:
        n = max(len(d1), len(d2))
        d1 = [1] * (n - len(d1)) + list(d1)
        d2 = [1] * (n - len(d2)) + list(d2)
        p = 1
        for x, y in zip(d1, d2):
            p *= max(int(x), int(y))
        return p

    batch = bcast_prod(a[:-2], b[:-2])
    return 2.0 * batch * m_ * k * n_


def _flops_gemm(node, shapes) -> Optional[float]:
    a = shapes.get(node.input[0])
    b = shapes.get(node.input[1])
    if a is None or b is None or len(a) != 2 or len(b) != 2:
        return None
    # Gemm 语义 Y = α·A·B (+β·C)。导出器常把权重存成 [out,in] 再用 transB=1
    # （Y = A@Bᵀ），此时 A[m,k]·B[out,in]ᵀ → 内维 k=in。不处理 transB 会把
    # [out,in] 误当 [k,n]，内维对不上而漏算（RT-DETR 47 个线性层全栽在这）。
    ma, ka = (a[0], a[1]) if not _attr(node, "transA", 0) else (a[1], a[0])
    kb, nb = (b[0], b[1]) if not _attr(node, "transB", 0) else (b[1], b[0])
    if ka != kb:
        return None
    return 2.0 * ma * ka * nb


def _graph_flops(nodes, shapes) -> tuple[float, int]:
    """遍历 FLOP 算子累加，返回 (flops, skipped)。"""
    flops = 0.0
    skipped = 0
    for node in nodes:
        op = node.op_type
        v: Optional[float] = None
        if op in ("Conv", "ConvTranspose"):
            v = _flops_conv(node, shapes)
        elif op == "MatMul":
            v = _flops_matmul(node, shapes)
        elif op == "Gemm":
            v = _flops_gemm(node, shapes)
        if v is None:
            if op in _FLOP_OPS:
                skipped += 1
        else:
            flops += v
    return flops, skipped


# ---------- 静态计数（保守：动态 Reshape 断链会漏） ----------

def onnx_gflops(onnx_path: str, img_size: int = 640) -> dict:
    """返回 {gflops: float, skipped: int}；skipped=形状拿不到而没算到的算子数。

    纯静态 shape_inference。RT-DETR 的 transformer Gemm 因动态 Reshape 断链
    会漏算，请优先用 onnx_gflops_runtime。
    """
    m = onnx.load(onnx_path)
    m = onnx.shape_inference.infer_shapes(m)
    shapes = _shape_map(m)
    shapes.update(_init_shapes(m))
    # 输入图像张量强制为 (1,3,imgsz,imgsz)：本实验固定 640；导出若含 dim_param 也能算
    for inp in m.graph.input:
        shapes[inp.name] = [1, 3, img_size, img_size]
    flops, skipped = _graph_flops(m.graph.node, shapes)
    return {"gflops": flops / 1e9, "skipped": skipped}


# ---------- 运行时计数（推荐：覆盖动态断链，640 下真值） ----------

def onnx_gflops_runtime(onnx_path: str, img_size: int = 640, threads: int = 16) -> dict:
    """返回 {gflops: float, skipped: int}。跑一次 ORT 前向，用真实中间形状计数。

    原理：静态推断拿不到的 FLOP 算子输入（RT-DETR 里是 102 个 Reshape 输出），
    临时声明为图输出，用 onnxruntime(CPU, 禁优化) 前向一次读出真实形状；
    权重形状直接取 initializer dims。覆盖动态 Reshape 断链 → skipped=0。
    开销 = 一次前向（RT-DETR-l ~几百 ms），不参与延迟计时。
    """
    import onnxruntime as ort

    m = onnx.shape_inference.infer_shapes(onnx.load(onnx_path))
    init_names = {i.name for i in m.graph.initializer}
    in_names = {i.name for i in m.graph.input}
    out_names = {o.name for o in m.graph.output}

    # 基线：静态可解的形状（含所有 initializer 权重）
    shapes = _shape_map(m)
    shapes.update(_init_shapes(m))
    for inp in m.graph.input:
        shapes[inp.name] = [1, 3, img_size, img_size]

    # 收集所有 FLOP 算子的激活输入（非权重/图入/图出）→ 运行时真值全覆盖。
    # 不止补「静态缺失」：某些静态形状被 dim_param→1 策略算错（transformer 动态维），
    # 也要用运行时形状覆盖，避免 Gemm 内维不匹配被误 skip。
    need: list[str] = []
    for node in m.graph.node:
        if node.op_type not in _FLOP_OPS:
            continue
        for i in node.input[:2]:
            if not i or i in init_names or i in in_names or i in out_names:
                continue
            if i not in need:
                need.append(i)
    if not need:
        flops, skipped = _graph_flops(m.graph.node, shapes)
        return {"gflops": flops / 1e9, "skipped": skipped}

    # 把需要运行时形状的张量声明为图输出（类型取 value_info，缺省 FLOAT；shape 留空）
    elem = {vi.name: vi.type.tensor_type.elem_type for vi in m.graph.value_info}
    for name in need:
        m.graph.output.append(
            onnx.helper.make_tensor_value_info(
                name, elem.get(name, onnx.TensorProto.FLOAT), None))

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = min(threads, 8)
    sess = ort.InferenceSession(
        m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    feed_name = m.graph.input[0].name
    outs = sess.run(None, {feed_name: np.zeros((1, 3, img_size, img_size), dtype=np.float32)})
    for name, arr in zip([o.name for o in sess.get_outputs()], outs):
        shapes[name] = list(arr.shape)  # 运行时真值覆盖

    flops, skipped = _graph_flops(m.graph.node, shapes)
    return {"gflops": flops / 1e9, "skipped": skipped}


# ---------- CLI ----------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runtime", action="store_true",
                    help="用运行时前向计数（覆盖动态断链，推荐）；默认静态")
    args = ap.parse_args()
    p = onnx_params_m(args.onnx)
    g = onnx_gflops_runtime(args.onnx, args.imgsz) if args.runtime \
        else onnx_gflops(args.onnx, args.imgsz)
    print(f"{args.onnx}: params={p:.2f}M  gflops={g['gflops']:.2f}G (skipped={g['skipped']})")


if __name__ == "__main__":
    main()
