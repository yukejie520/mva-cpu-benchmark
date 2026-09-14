"""S1-A 探针（2026-09-10）：ORT-QDQ 为什么只把少数 Conv 融成 QLinearConv？

背景（results/kernel_profile.csv）：naive static 的 YOLOv8n 优化执行图里
    conv_fp32=57, conv_int=7, conversion=636
即 64 个 Conv 里只有 7 个变成整数核，其余 57 个仍是 FP32，却额外背了 636 个转换算子。
而 quantize_int8.py 用的已经是 QuantFormat.QDQ + per_channel=True —— 所以问题不在"配置写错"，
而在**融合条件**。本探针把未融合的 Conv 逐个摊开，用可证伪的假设去定位根因。

三个竞争假设（都要能被数据否掉）：
  H1 分组卷积：Conv 的 group != 1 时 CPU EP 的 QLinearConv 不接（看 group 直方图）。
  H2 多消费者：Conv 输出被 >1 个下游节点消费（残差/Concat），激活的量化参数无法唯一 →
     QDQ 融合被拆开（看未融合 Conv 的多消费者比例）。
  H3 形状特殊：kernel 1x1 或 stride>1 的组合不被支持（看 kernel/stride 直方图）。

判定方式：分别统计"已融合"与"未融合"两组的属性分布。若某属性在未融合组里呈现
接近 100% 的一致性而在已融合组里不是，该假设成立；若两组分布相似，该假设被否掉。

用法：
    python scripts/probe_qdq_fusion.py --fp32 data/yolov8n.onnx \
        --quant data/yolov8n_static.onnx --label v8n_naive
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import onnx
import onnxruntime as ort

DEFAULT_THREADS = 16


def conv_attrs(path: str) -> dict[str, dict]:
    """源图里每个 Conv 节点的属性（group / kernel_shape / strides）。"""
    m = onnx.load(path)
    out = {}
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
        out[n.name] = {
            "group": int(a.get("group", 1)),
            "kernel": tuple(int(v) for v in a.get("kernel_shape", [])),
            "strides": tuple(int(v) for v in a.get("strides", [1, 1])),
            "inputs": list(n.input),
            "outputs": list(n.output),   # 查消费者必须用输出张量名，不是节点名
        }
    return out


def consumers(path: str) -> Counter:
    """张量名 → 消费者节点数（用于 H2 多消费者假设）。"""
    m = onnx.load(path)
    c: Counter = Counter()
    for n in m.graph.node:
        for t in n.input:
            c[t] += 1
    return c


def optimized_hist(path: str, threads: int = DEFAULT_THREADS,
                   optimized_path: str | None = None) -> tuple[Counter, dict[str, list[str]]]:
    """转储满优化执行图并返回 op 直方图；可选地保留优化图副本。

    ``threads`` 只改变会话配置，不改变源模型。``optimized_path`` 用于把
    S2 在第二个平台实际加载的优化图留档，避免只保存节点计数而丢失证据。
    """
    with tempfile.TemporaryDirectory() as td:
        opt = os.path.join(td, "opt.onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        so.optimized_model_filepath = opt
        sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        del sess
        m = onnx.load(opt)
        hist = Counter(n.op_type for n in m.graph.node)
        names = {n.op_type: [x.name for x in m.graph.node if x.op_type == n.op_type]
                 for n in m.graph.node}
        if optimized_path:
            destination = Path(optimized_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(opt, destination)
    return hist, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    src = conv_attrs(args.fp32)
    cons = consumers(args.fp32)
    hist, names = optimized_hist(args.quant)

    print(f"===== {args.label} =====")
    print(f"源 FP32 图 Conv 节点数: {len(src)}")
    print("优化执行图 op 直方图（前 12）:")
    for op, c in hist.most_common(12):
        print(f"    {op:28s} {c}")

    # ORT 融合时 QLinearConv 节点保留原 Conv 的名字 → 用名字做配对
    fused_names = set(names.get("QLinearConv", []))
    fp32_names = set(names.get("Conv", []))
    fused = [n for n in src if n in fused_names]
    unfused = [n for n in src if n in fp32_names]
    unknown = [n for n in src if n not in fused_names and n not in fp32_names]

    print(f"\n配对结果: 已融合 {len(fused)} / 未融合 {len(unfused)} / 名字对不上 {len(unknown)}")
    if unknown:
        print(f"  （对不上示例: {unknown[:3]}）")

    def dist(group: list[str], key: str) -> Counter:
        return Counter(
            (src[n]["kernel"] if key == "kernel" else src[n][key]) for n in group
        )

    for key in ("group", "kernel", "strides"):
        print(f"\n--- {key} 分布 ---")
        print(f"  {'已融合':>8}: {dict(dist(fused, key))}")
        print(f"  {'未融合':>8}: {dict(dist(unfused, key))}")

    # H2：未融合者里有多少 Conv 的输出被多个下游消费
    def multi_frac(group: list[str]) -> float:
        if not group:
            return float("nan")
        hit = sum(1 for n in group if cons.get(n, 0) > 1)
        return hit / len(group)

    print(f"\n--- H2 多消费者 ---")
    print(f"  已融合中输出被多消费者使用的比例: {multi_frac(fused):.1%}")
    print(f"  未融合中输出被多消费者使用的比例: {multi_frac(unfused):.1%}")

    # 未融合 Conv 的输入是否带 DQ（源量化图里的 QDQ 模式是否成形）
    q = onnx.load(args.quant)
    dq_out = {n.output[0] for n in q.graph.node if n.op_type == "DequantizeLinear"}
    has_dq_in = sum(1 for n in unfused if any(i in dq_out for i in src[n]["inputs"]))
    print(f"\n--- 未融合 Conv 的输入是否来自 DequantizeLinear ---")
    print(f"  {has_dq_in}/{len(unfused)} 的输入带 DQ（源图 QDQ 模式成形度）")

    # ---- H4：激活函数拆解（SiLU = Sigmoid + Mul）是否就是融合失败的原因 ----
    # ONNX 导出把 SiLU 写成 Sigmoid+Mul；若 Conv 的输出直接喂 Sigmoid，则
    # DQ->Conv->Q 之后紧跟的是浮点 Sigmoid，融合链被激活打断。
    m = onnx.load(args.fp32)
    consumer_op: dict[str, list[str]] = defaultdict(list)
    for n in m.graph.node:
        for t in n.input:
            consumer_op[t].append(n.op_type)

    def consumer_ops(node_name: str) -> tuple[str, ...]:
        """该 Conv 节点的**输出张量**被哪些算子消费（必须用输出张量名查）。"""
        ops = []
        for t in src[node_name]["outputs"]:
            ops.extend(consumer_op.get(t, []))
        return tuple(sorted(set(ops)))

    # ORT 给融合后的 QLinearConv 改了名，故"已融合"= 未出现在优化图 Conv 名单里的源 Conv
    fused_set = set(fused) | set(unknown)
    print(f"\n--- 源图中 Conv 输出的消费者算子组合（H4：激活拆解是否阻断融合）---")
    for tag, grp in (("已融合", sorted(fused_set)), ("未融合", sorted(unfused))):
        if not grp:
            print(f"  {tag}: 空")
            continue
        hist = Counter(consumer_ops(n) for n in grp)
        print(f"  {tag} {len(grp)} 个:")
        for combo, c in hist.most_common(6):
            print(f"      {c:3d} × {combo}")

    if fused_set:
        print(f"\n  被融合的 {len(fused_set)} 个 Conv 名（应为检测头 model.22）：")
        for n in sorted(fused_set):
            print(f"    {n}  kernel={src[n]['kernel']} strides={src[n]['strides']}")

    print("\n===== 结论 =====")
    print("若某个属性在'未融合'里接近 100% 而在'已融合'里不是 → 该假设成立。")


if __name__ == "__main__":
    main()
