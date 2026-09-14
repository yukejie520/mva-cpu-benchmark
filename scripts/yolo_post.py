"""YOLOv8 ONNX 输出的后处理（纯 numpy）——用于 e2e 延迟计时与检测结果。

关键事实（已实测，2026-09-05）：
- yolov8n.onnx 输出 output0 (1,84,8400)：前 4 = box(cx,cy,w,h)【已解码成 0-640 绝对像素】，
  后 80 = 类别分数【已含 sigmoid】。所以本模块不需要 stride 展开 / 再 sigmoid。
- RT-DETR ONNX 输出 (1,300,6) 端到端（含解码、免 NMS），不走本模块。

NMS 口径：conf>conf_thres 的候选按置信度降序逐框抑制同类的 IoU>iou_thres 框
（对齐 ultralytics 默认 iou=0.45 的单类 NMS；conf_thres=0.25 预测默认）。
"""
from __future__ import annotations

import numpy as np


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N,4) cxcywh -> xyxy。"""
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """单框 a 与一批框 b 的 IoU；a/b 均 (…,4) xyxy。"""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def yolov8_nms(raw: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45,
               max_det: int = 300, imgsz: int = 640) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对 (1,84,8400) 或 (8400,84) 的 YOLOv8 输出做解码+NMS。

    返回 (xyxy_boxes(N,4) float, conf(N,), cls_id(N,) int)，按置信度降序。
    NMS 的 box 坐标是模型已解码的 0-640 像素系（未映射回原图，计时用无需映射）。
    """
    if raw.ndim == 3:
        raw = raw[0]                      # (1,84,8400) -> (84,8400)
    raw = raw.T                           # (8400,84)
    scores = raw[:, 4:]                   # 已 sigmoid
    cls_id = np.argmax(scores, axis=1).astype(np.int32)
    conf = scores[np.arange(len(scores)), cls_id]
    keep = conf > conf_thres
    if not keep.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32)
    conf, cls_id = conf[keep], cls_id[keep]
    boxes = _xywh_to_xyxy(raw[keep, :4])

    order = np.argsort(-conf)
    conf, cls_id, boxes = conf[order], cls_id[order], boxes[order]

    selected_b, selected_c, selected_s = [], [], []
    suppressed = np.zeros(len(conf), dtype=bool)
    for i in range(len(conf)):
        if suppressed[i]:
            continue
        selected_s.append(conf[i]); selected_c.append(cls_id[i]); selected_b.append(boxes[i])
        if len(selected_b) >= max_det:
            break
        same_cls = (cls_id == cls_id[i]) & ~suppressed
        if same_cls.any():
            ious = _iou(boxes[i], boxes[same_cls])
            suppressed[same_cls] |= ious > iou_thres
            suppressed[i] = False  # 自己不能压自己（保持选中）
    return (np.stack(selected_b) if selected_b else np.zeros((0, 4)),
            np.array(selected_s) if selected_s else np.zeros(0),
            np.array(selected_c, dtype=np.int32) if selected_c else np.zeros(0, dtype=np.int32))
