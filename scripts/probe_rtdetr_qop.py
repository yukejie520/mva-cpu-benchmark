"""洞(c) 证伪实验（2026-09-10）：RT-DETR 的 static 失败，是不是和 YOLO 同一个根因？

背景
    YOLO 的 static 问题：图能建、能跑、scale 正常，只是 ORT 的 QDQ 优化器**融合不了**
    （QDQRegistry 缺 Sigmoid/Mul，ORT 1.28.0 registry.py:67-93）。
    RT-DETR 的 static 问题：**量化阶段就崩**——算 scale 时报 `scale=0/NaN`，
    输出退化成常数（300 个 query 的框≈0.004、conf≈0.006）。

    两者表面都是"static 不行"，但若机制陈述正确，它们**不是同一环节**。
    审稿人一定会问这个，所以要用实验把切割点钉死。

预测（可证伪）
    QDQ 与 QOperator 只改变**图的写出方式**（Q/DQ 对 vs QLinearConv），
    不改变**scale 的计算方式**（都在 onnxruntime.quantization.calibrate 里，
    与 quant_format 无关）。所以：RT-DETR 用 QOperator 应当**同样崩、崩在同一处**。
    → 若成立，"两者不是同一根因"这句话就是有证据的切割，而不是辩解。

对照
    同一份 data/rtdetr-l.onnx，同一校准集、同一 calib_n，只换 quant_format。
    两边的异常信息都原样捕获、不做任何美化，失败也是结果。

用法
    python scripts/probe_rtdetr_qop.py --model rtdetr-l
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_qdq_operator import model_input_name  # noqa: E402

CALIB = ROOT / "data" / "coco128" / "coco128" / "images" / "train2017"
DATA = ROOT / "data"

# Ultralytics RT-DETR 导出固定 640；用 letterbox 之外的最简缩放即可，
# 这里只要 scale 的统计分布，不追求检测精度。
SZ = 640
CALIB_N = 64


class _Reader:
    """ORT 校准读取器：get_next() 每次返回一个 dict，取完返回 None（不是生成器）。"""

    def __init__(self, input_name: str, n: int = CALIB_N, sz: int = SZ):
        self.input_name = input_name
        self.sz = sz
        self.paths = sorted(CALIB.glob("*.jpg"))[:n]
        self._i = 0
        self._tf = None

    def _transform(self):
        if self._tf is None:
            import torchvision.transforms as T
            self._tf = T.Compose([T.Resize((self.sz, self.sz)), T.ToTensor()])
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


def try_quantize(src: Path, dst: Path, fmt: str, input_name: str) -> dict:
    """跑一次 static 量化，把成功/失败都原样记录。fmt ∈ {'QDQ','QOperator'}。"""
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    qf = {"QDQ": QuantFormat.QDQ, "QOperator": QuantFormat.QOperator}[fmt]
    # QOperator 的 QLinearConv 规定激活为 uint8；QDQ 下用 int8（与正文 YOLO 一致）
    act = QuantType.QUInt8 if fmt == "QOperator" else QuantType.QInt8
    rec = {"format": fmt, "ok": False, "error_type": "", "error_msg": "", "wrote_file": False}
    try:
        quantize_static(str(src), str(dst), _Reader(input_name),
                        quant_format=qf, per_channel=True,
                        activation_type=act, weight_type=QuantType.QInt8)
        rec["ok"] = True
        rec["wrote_file"] = dst.exists()
    except Exception as e:  # 失败也是结果，原样捕获
        rec["error_type"] = type(e).__name__
        rec["error_msg"] = " | ".join(str(e).splitlines())[:600]
        rec["trace_tail"] = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])[:800]
    return rec


def run_smoke(path: Path, input_name: str, n: int = 3) -> dict:
    """前向跑几张图，看输出是不是退化（全常数 / NaN / 全零）。"""
    import onnxruntime as ort
    import torchvision.transforms as T
    from PIL import Image
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    t = T.Compose([T.Resize((SZ, SZ)), T.ToTensor()])
    outs = []
    for p in sorted(CALIB.glob("*.jpg"))[:n]:
        x = t(Image.open(p).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
        outs.append(np.asarray(sess.run(None, {input_name: x})[0]))
    o = np.concatenate([a.reshape(-1) for a in outs])
    return {
        "finite": bool(np.isfinite(o).all()),
        "std": float(np.nanstd(o)),
        "min": float(np.nanmin(o)), "max": float(np.nanmax(o)),
        "is_constant": bool(np.isfinite(o).all() and np.nanstd(o) < 1e-8),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rtdetr-l")
    args = ap.parse_args()

    src = DATA / f"{args.model}.onnx"
    if not src.exists():
        raise SystemExit(f"缺 {src}")
    input_name, shape = model_input_name(str(src))
    print(f"[源] {src.name}  输入={input_name}{shape}  校准={CALIB_N} 张 @ {SZ}\n")

    results = []
    for fmt in ("QDQ", "QOperator"):
        dst = DATA / f"{args.model}_static{'_qop' if fmt == 'QOperator' else ''}.onnx"
        print(f"===== 尝试 {fmt} =====", flush=True)
        r = try_quantize(src, dst, fmt, input_name)
        if r["ok"]:
            print(f"  量化成功，已写出 {dst.name}（{dst.stat().st_size/1e6:.1f} MB）")
            try:
                s = run_smoke(dst, input_name)
                r["smoke"] = s
                flag = "退化(常数)" if s["is_constant"] else ("非有限" if not s["finite"] else "正常")
                print(f"  前向冒烟：{flag}  std={s['std']:.3e}  range=[{s['min']:.4g}, {s['max']:.4g}]")
            except Exception as e:
                r["smoke_error"] = f"{type(e).__name__}: {e}"
                print(f"  前向冒烟失败：{r['smoke_error'][:200]}")
        else:
            print(f"  量化失败：{r['error_type']}")
            print(f"    {r['error_msg']}")
        results.append(r)
        print()

    print("===== 判决 =====")
    for r in results:
        if r["ok"]:
            sm = r.get("smoke", {})
            verdict = "退化" if sm.get("is_constant") else "可跑"
            print(f"  {r['format']:10s} 量化成功 → 前向 {verdict}")
        else:
            print(f"  {r['format']:10s} 量化阶段即失败 → {r['error_type']}: {r['error_msg'][:120]}")
    same = (results[0]["ok"] == results[1]["ok"])
    print(f"\n  两种格式结局是否相同：{'是' if same else '否 —— 说明格式影响了 scale 计算，机制陈述需修正'}")


if __name__ == "__main__":
    main()
