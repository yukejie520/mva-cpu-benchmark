"""测量公共库：所有延迟测量脚本共享的预处理/统计/ORT 会话。

为什么抽出来：YOLO 和 RT-DETR 两族必须用**同一份** letterbox 预处理、
同一种统计口径、同一个 ORT 会话配置——否则跨族对比就不公平。
线程数(16)是用户拍板的全局协议，任何脚本不得私自改。

协议（2026-09-05 拍板）：
- 台架：本地 Intel i7-14650HX，16 线程（intra_op）
- 输入：真实图 letterbox 到 640，归一化 /255（对齐 ultralytics）
- 延迟：取中位（防离群慢值污染）
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_THREADS = 16  # 用户 2026-09-05 拍板，全篇唯一
IMGSZ = 640


def aggregate(times: list[float]) -> dict:
    """把若干次延迟(ms)聚合成统计量；fps 用中位（防离群慢值）。times 空则抛错。"""
    if not times:
        raise ValueError("times must be non-empty")
    arr = np.array(times, dtype=np.float64)
    med = float(np.median(arr))
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": med,
        "min_ms": float(arr.min()),
        "p90_ms": float(np.percentile(arr, 90)),
        "std_ms": float(arr.std()),
        "fps": 1000.0 / med if med > 0 else float("inf"),
    }


def letterbox(im: np.ndarray, new_shape: int = IMGSZ, color: int = 114) -> np.ndarray:
    """等比缩放 + 灰边居中 pad（对齐 ultralytics 默认）。"""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (new_shape - new_w) / 2, (new_shape - new_h) / 2
    im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color)
    )


def preprocess(path: str, imgsz: int = IMGSZ) -> np.ndarray:
    """读图 -> letterbox -> BGR2RGB -> /255 -> CHW -> 1x3xHxW float32。"""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    img = letterbox(img, imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(img, 0)


def build_session(onnx_path: str, threads: int = DEFAULT_THREADS) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
    )


def list_images(img_dir: str, n: int | None = None) -> list[Path]:
    """按文件名排序取 jpg（确定性顺序，可复现）。"""
    paths = sorted(Path(img_dir).rglob("*.jpg"))
    return paths[:n] if n else paths


def measure_forward(sess: ort.InferenceSession, inputs: list[np.ndarray],
                    warmup: int = 20, reps: int = 100) -> list[float]:
    """对多张预处理输入测前向延迟(ms)；warmup 用第一张重复跑（不计时）。"""
    inp = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    for _ in range(warmup):
        sess.run(outs, {inp: inputs[0]})
    times: list[float] = []
    for x in inputs:
        for _ in range(reps):
            t0 = time.perf_counter()
            sess.run(outs, {inp: x})
            times.append((time.perf_counter() - t0) * 1e3)
    return times
