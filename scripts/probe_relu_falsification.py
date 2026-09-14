"""洞(b) 证伪实验（2026-09-10）：SiLU 不融合的机制，是"相关"还是"可预测"？

问题来源
    我们已经知道：YOLOv8n 的 QDQ 量化只有 7/64 个 Conv 融合成 QLinearConv。
    给出的机制是"PyTorch 把 SiLU 导成 Sigmoid + Mul，而 ORT 的 QDQRegistry 里
    没有 Sigmoid / Mul（ORT 1.28.0，quantization/registry.py:67-93）"。

    但这句话如果只是"我们观察到的现象"，它就是**相关**；如果它能**预测**没见过的
    情况，它才是**机制**。这个脚本就是做那个预测检验。

预测（来自 registry.py 第 72 行）
    QDQRegistry 里 **有 Relu**（`"Relu": QDQRemovableActivation`），**没有 Sigmoid / Mul**。
    所以：同一拓扑、只把激活函数从 SiLU 换成 ReLU，QDQ 融合率应当从接近 0 跳到接近 100%。

设计（两个层次，缺一不可）
    层一 · 单因子对照（干净的因果）：同一个自制小 CNN，只换激活。
        relu 版 vs silu 版，其余（层数、通道、BN、输入形状、opset、校准集、量化参数）全同。
    层二 · 真实模型（生态效度）：ResNet-18（全 ReLU 系）vs EfficientNet-B0（全 SiLU 系）。
        证明结论不是自制玩具特有的。

先决校验（不通过就 abort，不产出误导结论）
    打印两个 FP32 图的算子直方图，确认 ReLU 真的导成 `Relu`、SiLU 真的导成 `Sigmoid`+`Mul`。
    若 PyTorch 版本把它导成别的形状，机制陈述就要跟着改——这一步就是为了抓那种情况。

用法
    python scripts/probe_relu_falsification.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_qdq_operator import census, model_input_name  # noqa: E402

CALIB = ROOT / "data" / "coco128" / "coco128" / "images" / "train2017"
DATA = ROOT / "data"
OUT_DIR = ROOT / "data" / "falsify"
OUT_CSV = ROOT / "results" / "probe_relu_falsification.csv"

SZ = 224
CALIB_N = 64
OPSET = 17


def _registry_ops(var: str) -> frozenset:
    """从 ORT 源码里现读注册表，避免把版本相关的算子表抄死在脚本里。

    这是**可核查出处**：onnxruntime/quantization/registry.py，ORT 1.28.0
    （QDQRegistry 在 67-93 行，QLinearOpsRegistry 在 40-64 行）。
    """
    import onnxruntime, pathlib, re
    src = (pathlib.Path(onnxruntime.__file__).parent / "quantization" / "registry.py")
    text = src.read_text(encoding="utf-8")
    start = text.index(f"{var} = {{")
    body = text[start:text.index("}", start)]
    return frozenset(re.findall(r'"(\w+)"\s*:', body))


QDQ_REG = _registry_ops("QDQRegistry")           # 25 个（含 Relu，不含 Sigmoid/Mul/Add）
QOP_REG = _registry_ops("QLinearOpsRegistry")    # 23 个
# 关键差集：QOperator 模式能处理、QDQ 模式没有的算子 —— 就是量化域的断点
QOP_ONLY = sorted(QOP_REG - QDQ_REG)


# ---------------------------------------------------------------- 模型定义

def tiny_cnn(act: str):
    """单因子对照网：4 个 Conv+BN+激活块，只换激活函数。

    故意不用残差/池化等会引入额外算子的结构，让"唯一变量"真的唯一。
    """
    import torch.nn as nn
    a = {"relu": nn.ReLU, "silu": nn.SiLU}[act]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [3, 16, 32, 32, 48]
            self.blocks = nn.Sequential(*[
                nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 3, padding=1, bias=False),
                              nn.BatchNorm2d(ch[i + 1]), a())
                for i in range(len(ch) - 1)
            ])
            self.head = nn.Conv2d(ch[-1], 8, 1)

        def forward(self, x):
            return self.head(self.blocks(x))

    return Net().eval()


def real_model(name: str):
    """真实模型：resnet18 = 全 ReLU 系；efficientnet_b0 = 全 SiLU 系。"""
    import torch
    from torchvision import models
    fn = {"resnet18": models.resnet18, "efficientnet_b0": models.efficientnet_b0}[name]
    torch.manual_seed(0)
    return fn(weights=None).eval()  # 不下载预训练权重，离线可跑


# ---------------------------------------------------------------- 导出与量化

class _Reader:
    """ORT 校准读取器：从 data/coco128 抽 N 张图，缩放到 SZ×SZ，NCHW float32。

    注意 ORT 的 CalibrationDataReader 协议：get_next() 每次调用返回**一个 dict**，
    数据取完返回 None（不是返回生成器——那是初学者最容易踩的坑，ORT 会报
    `'generator' object has no attribute 'keys'`）。
    """

    def __init__(self, input_name: str, n: int = CALIB_N):
        self.input_name = input_name
        self.paths = sorted(CALIB.glob("*.jpg"))[:n]
        self._i = 0
        self._tf = None

    def _transform(self):
        if self._tf is None:
            import torchvision.transforms as T
            self._tf = T.Compose([T.Resize((SZ, SZ)), T.ToTensor()])
        return self._tf

    def get_next(self):
        from PIL import Image
        if self._i >= len(self.paths):
            return None
        p = self.paths[self._i]
        self._i += 1
        x = self._transform()(Image.open(p).convert("RGB")).unsqueeze(0)
        return {self.input_name: x.numpy().astype(np.float32)}

    def rewind(self):
        self._i = 0


def export_onnx(model, dst: Path) -> Path:
    """用旧版（TorchScript）导出器。

    必须 dynamo=False：新版 dynamo 导出器需要 onnxscript，且会把 SiLU 导成单个融合算子，
    那样就复现不出正文 YOLO 图里的 Sigmoid+Mul 形状，实验失去可比性。
    """
    import torch
    x = torch.randn(1, 3, SZ, SZ)
    torch.onnx.export(model, (x,), str(dst), opset_version=OPSET,
                      input_names=["images"], output_names=["output"],
                      do_constant_folding=True, dynamo=False)
    return dst


def op_hist(path: Path) -> dict:
    """FP32 原始图（未优化）的算子直方图——先决校验用。"""
    import onnx
    m = onnx.load(str(path))
    h: dict = {}
    for n in m.graph.node:
        h[n.op_type] = h.get(n.op_type, 0) + 1
    return h


def blocker_profile(fp32_path: Path) -> dict:
    """按注册表差异，**预测**这个图里有多少卷积应当融合。

    规则（由 registry.py 的差集推出，不是拟合）：
        一个卷积要融合成 QLinearConv，它的每个消费算子都得有 QDQ 量化器；
        只要有一个消费算子落在"QOperator 有、QDQ 没有"的 8 个算子集合里，
        量化域就在那里断掉，该卷积退回 FP32。

    返回：blocker 算子的出现次数 + 预测"可融合"的卷积数。
    """
    import onnx
    m = onnx.load(str(fp32_path))
    consumers: dict = {}
    for n in m.graph.node:
        for t in n.input:
            consumers.setdefault(t, []).append(n.op_type)
    out_names = {o.name for o in m.graph.output}

    BLOCKERS = set(QOP_ONLY)  # 8 个：QOperator 有处理器、QDQ 没有
    blockers, predicted = {}, 0
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        # 同名输入会被算进 consumers；只统计**不在** QDQ 注册表里的消费算子
        cons = set()
        for t in n.output:
            if t in out_names:
                continue  # 图输出不构成阻断
            cons |= {c for c in consumers.get(t, []) if c not in QDQ_REG}
        if not cons:
            predicted += 1
        else:
            for c in cons:
                blockers[c] = blockers.get(c, 0) + 1
    return {"blocker_ops": blockers, "predicted_fusable": predicted}


def quantize_qdq(src: Path, dst: Path) -> Path:
    """QDQ 静态量化（与正文 YOLO 用的同一函数、同一参数）。"""
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    name, _shape = model_input_name(str(src))
    quantize_static(str(src), str(dst), _Reader(name),
                    quant_format=QuantFormat.QDQ,
                    per_channel=True,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8)
    return dst


# ---------------------------------------------------------------- 主流程

CASES = [
    ("tiny_relu", "单因子对照·自制网 ReLU", lambda: tiny_cnn("relu")),
    ("tiny_silu", "单因子对照·自制网 SiLU", lambda: tiny_cnn("silu")),
    ("resnet18", "真实模型·ResNet-18 (ReLU 系)", lambda: real_model("resnet18")),
    ("efficientnet_b0", "真实模型·EfficientNet-B0 (SiLU 系)", lambda: real_model("efficientnet_b0")),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_lists = [n for n in CALIB.glob("*.jpg")]
    if len(fp32_lists) < 8:
        raise SystemExit(f"校准图不足：{CALIB} 只有 {len(fp32_lists)} 张")

    rows = []
    for tag, label, build in CASES:
        fp32 = OUT_DIR / f"{tag}.onnx"
        qdq = OUT_DIR / f"{tag}_static.onnx"
        print(f"\n===== {label}  ({tag}) =====", flush=True)

        if not fp32.exists():
            export_onnx(build(), fp32)
        raw = op_hist(fp32)
        conv_fp32_raw = raw.get("Conv", 0)
        acts = {k: raw.get(k, 0) for k in ("Relu", "Sigmoid", "Mul", "Hardswish", "Silu")}
        print(f"  [先决校验] FP32 图 Conv={conv_fp32_raw}  激活算子={acts}")
        if conv_fp32_raw == 0:
            raise SystemExit(f"{tag} 导出图里没有 Conv，实验无效")

        prof = blocker_profile(fp32)
        print(f"  [预测] 按注册表差集，可融合卷积 = {prof['predicted_fusable']}/{conv_fp32_raw}"
              f"   断点算子={prof['blocker_ops'] or '无'}")

        if not qdq.exists():
            quantize_qdq(fp32, qdq)
        c = census(qdq)
        fuse_rate = c["conv_int"] / conv_fp32_raw if conv_fp32_raw else float("nan")
        print(f"  [量化后] QLinearConv={c['QLinearConv']}  Conv(FP32 残留)={c['conv_fp32']}  "
              f"Q/DQ 转换={c['conversion']}  总节点={c['total_nodes']}")
        hit = "命中" if c["conv_int"] == prof["predicted_fusable"] else \
              f"偏差 {c['conv_int'] - prof['predicted_fusable']:+d}"
        print(f"  [融合率] {c['conv_int']}/{conv_fp32_raw} = {fuse_rate:.0%}   预测校验：{hit}")

        rows.append({
            "case": tag, "label": label, "conv_in_fp32_graph": conv_fp32_raw,
            "act_relu": acts["Relu"], "act_sigmoid": acts["Sigmoid"], "act_mul": acts["Mul"],
            "blocker_ops": ";".join(f"{k}x{v}" for k, v in sorted(prof["blocker_ops"].items())),
            "predicted_fusable": prof["predicted_fusable"],
            "qlinearconv": c["QLinearConv"], "conv_fp32_left": c["conv_fp32"],
            "conversion_nodes": c["conversion"], "total_nodes": c["total_nodes"],
            "fuse_rate": round(fuse_rate, 4),
        })

    # 参考组：论文主角的两个 YOLO 模型（复用已有产物，不重量化），
    # 让"预测规则"在真实被测对象上也被检验一次。
    print(f"\n===== 参考组：被测 YOLO 模型（复用已有 QDQ 产物）=====")
    for tag in ("yolov8n", "yolov8s"):
        fp32, qdq = DATA / f"{tag}.onnx", DATA / f"{tag}_static.onnx"
        if not (fp32.exists() and qdq.exists()):
            print(f"  [skip] 缺 {tag} 产物")
            continue
        prof = blocker_profile(fp32)
        raw = op_hist(fp32)
        c = census(qdq)
        fusable = c["QLinearConv"]
        print(f"  {tag}: FP32 图 Conv={raw.get('Conv', 0)}  "
              f"Sigmoid={raw.get('Sigmoid', 0)} Mul={raw.get('Mul', 0)} Add={raw.get('Add', 0)}")
        print(f"    预测可融合={prof['predicted_fusable']}  实测={fusable}  "
              f"断点算子={prof['blocker_ops'] or '无'}")
        rows.append({
            "case": tag, "label": f"被测模型 {tag}", "conv_in_fp32_graph": raw.get("Conv", 0),
            "act_relu": raw.get("Relu", 0), "act_sigmoid": raw.get("Sigmoid", 0),
            "act_mul": raw.get("Mul", 0),
            "blocker_ops": ";".join(f"{k}x{v}" for k, v in sorted(prof["blocker_ops"].items())),
            "predicted_fusable": prof["predicted_fusable"],
            "qlinearconv": fusable, "conv_fp32_left": c["conv_fp32"],
            "conversion_nodes": c["conversion"], "total_nodes": c["total_nodes"],
            "fuse_rate": round(fusable / raw.get("Conv", 1), 4),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n===== 判决 =====")
    for r in rows:
        print(f"  {r['case']:20s} 融合 {r['qlinearconv']:3d}/{r['conv_in_fp32_graph']:<3d} "
              f"= {r['fuse_rate']:>6.1%}   (ReLU={r['act_relu']} Sigmoid={r['act_sigmoid']} "
              f"Mul={r['act_mul']})")
    print(f"\n[csv] -> {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
