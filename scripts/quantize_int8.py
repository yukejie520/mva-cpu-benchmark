"""INT8 量化（ONNX Runtime）：dynamic / static 两方案，跨族对比敏感度用。

用途：账本 §3。给 6 个 FP32 ONNX 产出 INT8 ONNX，供 ①同窗口延迟测量、②子集 mAPΔ。

两方案区别（通俗版）：
- Dynamic：只有**权重**压成 INT8，激活(中间结果)每次前向临时量化。无需校准数据、一次成型，
  对 CPU 快一点但加速有限；精度损失通常小。写出来的是 QOperator 格式。
- Static：权重+激活都用校准集统计出的**固定**缩放系数压成 INT8，推理最省（激活不重复量化）。
  需要跑一遍校准集（我们用本地 coco128）。写出来的是 QDQ 格式（可感知量化）。
依赖 onnxruntime.quantization（已确认 1.28.0 可用）。

用法：
    python scripts/quantize_int8.py --model data/yolov8n.onnx --out data/yolov8n_dyn.onnx \
        --scheme dynamic
    python scripts/quantize_int8.py --model data/yolov8n.onnx --out data/yolov8n_static.onnx \
        --scheme static --calib data/coco128/coco128/images/train2017 --calib-n 64
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from eval_common import IMGSZ, preprocess  # noqa: E402

DYNAMIC_OP_TYPES = ["Conv", "MatMul", "Gemm"]  # 只量化这些核，避免破坏其余算子


class _CalibReader:
    """ORT 1.28 的 CalibrationDataReader 约定：实现 get_next()，耗尽返回 None。"""

    def __init__(self, img_dir: str, input_name: str, n: int):
        from eval_common import list_images

        self.input_name = input_name
        self.paths = list_images(img_dir, n)
        self.idx = 0

    def get_next(self) -> dict | None:
        if self.idx >= len(self.paths):
            return None
        x = preprocess(str(self.paths[self.idx]), IMGSZ)
        self.idx += 1
        return {self.input_name: x}


def model_input_name(model_path: str) -> tuple[str, list[int]]:
    """读 ONNX 输入名与维度（不建 ORT 会话，轻量）。返回 (name, shape)。"""
    import onnx

    m = onnx.load(model_path)
    vi = m.graph.input[0]
    shape = [d.dim_value if d.dim_value else d.dim_param for d in vi.type.tensor_type.shape.dim]
    return vi.name, [int(s) if isinstance(s, int) else s for s in shape]


def _ensure_parent(out: str) -> None:
    """量化库写产物前先建好父目录，避免因调用方 cwd 不同而失败。"""
    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)


def quantize_dynamic_onnx(model: str, out: str) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    _ensure_parent(out)
    quantize_dynamic(model, out, weight_type=QuantType.QInt8, op_types_to_quantize=DYNAMIC_OP_TYPES)
    return Path(out)


def _output_concat_nodes(model_path: str) -> list[str]:
    """找「输出拼装区」的 Concat 节点名，供 static 量化排除。

    为什么（2026-09-06 实测发现，账本 §3 拟作贡献点）：
    YOLOv8/RT-DETR 的最终输出把「框坐标(0~640 大数)」与「分类分数(0~1 小数，
    sigmoid 输出)」用 Concat 拼成一束。static 量化会把该 Concat 也量化，缩放系数
    由大数(640)决定 → 小数那路被压成 0 → 检不出任何框（mAP≈0）。排除这些 Concat
    让输出拼接保持 FP32，即可恢复。

    怎么找：从 graph output 沿「生产者」反向往上游走，遇到 Concat 就收下；但每
    条分支一旦遇到 Conv/MatMul/Gemm（= 量化边界，头卷积在更下游）就停。这样只收
    到「紧贴输出的拼装区」的 Concat（YOLO 的 Concat_2/3、RT-DETR 的 Concat_8），
    不会误收骨干 C2f 里深层的 Concat。
    """
    import onnx

    m = onnx.load(model_path)
    out2node = {}
    for n in m.graph.node:
        for o in n.output:
            out2node[o] = n
    WEIGHTED = {"Conv", "MatMul", "Gemm"}  # 遇到就停：量化边界
    found: list[str] = []
    seen: set[str] = set()
    frontier = [o.name for o in m.graph.output]
    while frontier:
        tensor = frontier.pop()
        if tensor in seen:
            continue
        seen.add(tensor)
        n = out2node.get(tensor)
        if n is None:
            continue
        if n.op_type == "Concat":
            found.append(n.name)
        if n.op_type in WEIGHTED:
            continue  # 不再向该分支更上游深入
        frontier.extend(n.input)
    return list(dict.fromkeys(f for f in found if f))


def _output_concat_module(model_path: str) -> str | None:
    """返回产出 graph output 的 Concat 所在模块前缀（如 YOLOv8 的 '/model.22/'）。

    用途：static 量化里「检测头整体保留 FP32」的 recipe——把输出 Concat 所在的整个
    模块（YOLOv8 的 Detect 头 model.22，内含 box/class 分支全部卷积）排除出量化。
    为什么（2026-09-06 实测，账本 §3 发现①延伸）：只排除输出 Concat 虽救回分类，
    但检测头卷积仍被量化，YOLOv8 子集 mAP 掉 ~17%；把整个 Detect 头留 FP32 后
    Δ≈-1%（精度无损）。头是小模块，体积仍 ~FP32 的一半。
    """
    concats = _output_concat_nodes(model_path)
    if not concats:
        return None
    first = concats[0]  # 形如 /model.22/Concat_3
    slash = first.rfind("/")
    if slash <= 0:
        return None
    return first[: slash + 1]  # '/model.22/'


def _exclude_list(model_path: str, exclude_output_concat: bool = True,
                  exclude_head_module: bool = False) -> list[str]:
    """组装 static 量化的排除节点清单（纯逻辑，便于单测）。

    exclude_output_concat：排除产出 graph output 的 Concat（否则分类被压平）。
    exclude_head_module：再把输出 Concat 所在整模块（YOLOv8 Detect 头）也排除，
    即「检测头保留 FP32、骨干/颈部仍 INT8」的 selective 配方。
    """
    nodes_to_exclude = _output_concat_nodes(model_path) if exclude_output_concat else []
    if exclude_head_module:
        module = _output_concat_module(model_path)
        if module:  # 找不到输出 Concat 就不整头排除（退化为仅排除 Concat）
            import onnx
            m = onnx.load(model_path)
            nodes_to_exclude += [n.name for n in m.graph.node
                                 if n.name.startswith(module) and n.name not in nodes_to_exclude]
    return nodes_to_exclude


def quantize_static_onnx(model: str, out: str, calib_dir: str, calib_n: int = 64,
                         exclude_output_concat: bool = True,
                         exclude_head_module: bool = False,
                         quant_format: str = "QDQ",
                         activation_type: str = "QInt8") -> Path:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    _ensure_parent(out)
    name, shape = model_input_name(model)
    if shape and shape[0] not in (1, -1, None):
        raise ValueError(f"输入 batch 维不是 1：{name}{shape}")
    reader = _CalibReader(calib_dir, name, calib_n)
    if not reader.paths:
        raise FileNotFoundError(f"{calib_dir} 下没有校准图")
    nodes_to_exclude = _exclude_list(model, exclude_output_concat, exclude_head_module)
    quantize_static(
        model, out, reader,
        quant_format=getattr(QuantFormat, quant_format),
        per_channel=True,
        activation_type=getattr(QuantType, activation_type),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,
    )
    return Path(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scheme", choices=["dynamic", "static"], required=True)
    ap.add_argument("--calib", default=None, help="static 校准图目录")
    ap.add_argument("--calib-n", type=int, default=64)
    ap.add_argument("--format", choices=["QDQ", "QOperator"], default="QDQ",
                    help="static graph representation; report activation type alongside it")
    ap.add_argument("--activation", choices=["QInt8", "QUInt8"], default="QInt8")
    args = ap.parse_args()

    if args.scheme == "dynamic":
        out = quantize_dynamic_onnx(args.model, args.out)
    else:
        if not args.calib:
            raise SystemExit("static 需要 --calib 校准图目录")
        out = quantize_static_onnx(args.model, args.out, args.calib, args.calib_n,
                                   quant_format=args.format,
                                   activation_type=args.activation)
    size_in = os.path.getsize(args.model)
    size_out = out.stat().st_size
    print(f"[quant] {Path(args.model).name} -> {out.name}  {size_in/1e6:.1f}MB -> {size_out/1e6:.1f}MB"
          f" ({size_out/size_in*100:.0f}%)")
    # 轻量验证：能建会话并前向一次
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, IMGSZ, IMGSZ), np.float32)
    o = sess.run(None, {name: dummy})[0]
    print(f"[ok] 前向成功 output={o.shape}")


if __name__ == "__main__":
    main()
