"""轻量 NumPy 检查用的本地检测 AP 评估库（纯 CPU）。

背景：这个模块用于没有标准 COCO API 时的开发阶段相对比较。它近似 Ultralytics
的匹配和 PR 插值，但**不是**官方 COCOeval：它不处理 crowd-ignore、每图 maxDets
和 COCOeval 的面积分层。因此它的输出不能在论文中标为标准 COCO mAP。
此前将它称为官方口径是不准确的，现已明确降级为探索性指标。

匹配实现的参考行为：
1) match_predictions：逐 IoU 阈值(0.5:0.95 步 0.05 共 10 档)，对「同类」的
   pred×gt IoU 矩阵做贪心去重（按 IoU 降序，pred 与 gt 各最多匹配一次）→ 每预测一行 10 列 bool tp。
2) ap_per_class：按置信度累计 precision-recall 求各 IoU 下 AP → 平均得 mAP50-95。
本模块的 match_tp 复用了第 1 步的近似实现，AP 聚合采用 101 点插值。它保留用于
FP32/INT8 的开发阶段同管线相对差值；投稿用绝对精度请运行
``scripts/evaluate_full_coco_official.py``。

坐标系约定：所有预测框/真值框统一换算到**原图像素 xyxy** 后再匹配（IoU 匹配
与坐标系无关，只需双方同空间；这样避免纠缠 letterbox pad）。
"""
from __future__ import annotations

import numpy as np

# COCO 80 类：image_id 在 instances_val2017 里的 category_id 是 1..90 的稀疏编号，
# 需映射到 Ultralytics 训练用的 0..79 连续索引。顺序=官方 categories 文件。
COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
_CAT_TO_IDX = {cid: i for i, cid in enumerate(COCO_CAT_IDS)}

IOUV = np.linspace(0.5, 0.95, 10)  # COCO 十个 IoU 档


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """开发阶段 101 点插值 AP；不等同于完整 COCOeval。"""
    mrec = np.concatenate(([0.0], np.asarray(recall, dtype=np.float64), [1.0]))
    mpre = np.concatenate(([1.0], np.asarray(precision, dtype=np.float64), [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def _smooth(y: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Small moving-average smoother used only to choose the F1 operating point."""
    y = np.asarray(y, dtype=np.float64)
    if y.size < 3:
        return y
    window = max(3, int(round(y.size * fraction * 2)))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _ap_per_class_numpy(tp: np.ndarray, conf: np.ndarray,
                        pred_cls: np.ndarray, target_cls: np.ndarray):
    """Return AP arrays needed by ``_finish_map`` without external packages."""
    order = np.argsort(-conf, kind="stable")
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    unique_classes, target_counts = np.unique(target_cls, return_counts=True)
    nc, niou = unique_classes.size, tp.shape[1]
    px = np.linspace(0.0, 1.0, 1000)
    p_curve = np.zeros((nc, px.size), dtype=np.float64)
    r_curve = np.zeros((nc, px.size), dtype=np.float64)
    ap = np.zeros((nc, niou), dtype=np.float64)
    eps = 1e-16

    for ci, cls in enumerate(unique_classes):
        mask = pred_cls == cls
        n_pred = int(mask.sum())
        n_gt = int(target_counts[ci])
        if n_pred == 0 or n_gt == 0:
            continue
        fpc = (1.0 - tp[mask]).cumsum(axis=0)
        tpc = tp[mask].cumsum(axis=0)
        recall = tpc / (n_gt + eps)
        precision = tpc / (tpc + fpc + eps)
        r_curve[ci] = np.interp(-px, -conf[mask], recall[:, 0], left=0.0)
        p_curve[ci] = np.interp(-px, -conf[mask], precision[:, 0], left=1.0)
        for j in range(niou):
            ap[ci, j] = _compute_ap(recall[:, j], precision[:, j])

    f1_curve = 2.0 * p_curve * r_curve / (p_curve + r_curve + eps)
    best = int(_smooth(f1_curve.mean(axis=0), 0.1).argmax())
    return p_curve[:, best], r_curve[:, best], f1_curve[:, best], ap, unique_classes.astype(np.int64)


def category_to_idx(category_id: int) -> int:
    """COCO category_id(1..90) -> 0..79 连续索引。"""
    try:
        return _CAT_TO_IDX[category_id]
    except KeyError:
        raise ValueError(f"未知 COCO category_id: {category_id}") from None


def xywh_to_xyxy(bboxes: np.ndarray) -> np.ndarray:
    """(N,4) COCO xywh（左上角+宽高）-> xyxy（绝对像素）。"""
    out = np.asarray(bboxes, dtype=np.float64).copy()
    out[:, 2] += out[:, 0]  # x2 = x + w
    out[:, 3] += out[:, 1]  # y2 = y + h
    return out


def box_iou(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """真值(M,4)xyxy × 预测(N,4)xyxy -> IoU (M,N)。"""
    gt = np.asarray(gt, dtype=np.float64)
    pr = np.asarray(pr, dtype=np.float64)
    x1 = np.maximum(gt[:, None, 0], pr[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], pr[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], pr[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], pr[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    ap = (pr[:, 2] - pr[:, 0]) * (pr[:, 3] - pr[:, 1])
    union = ag[:, None] + ap[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match_tp(pred_cls: np.ndarray, gt_cls: np.ndarray, iou_mat: np.ndarray,
             iouv: np.ndarray = IOUV) -> np.ndarray:
    """近似 ultralytics DetectionValidator.match_predictions 的 NumPy 实现。

    参数：pred_cls(N,) 预测类；gt_cls(M,) 真值类；iou_mat(M,N) 逐对 IoU。
    返回：(N,10) bool，第 j 列 = 该预测在 IoU>=iouv[j] 且类别匹配下是否被贪心匹配上。
    关键差异点：不等类 IoU 先清零，再按 IoU 降序对 pred 轴和 gt 轴各去重一次。
    """
    pred_cls = np.asarray(pred_cls)
    gt_cls = np.asarray(gt_cls)
    iou_mat = np.asarray(iou_mat, dtype=np.float64)
    n_pred = pred_cls.shape[0]
    n_thr = len(iouv)
    correct = np.zeros((n_pred, n_thr), dtype=bool)
    if n_pred == 0 or gt_cls.shape[0] == 0:
        return correct
    same_cls = gt_cls[:, None] == pred_cls[None, :]  # (M,N)
    iou_same = iou_mat * same_cls                    # 不等类清零
    for j, thr in enumerate(iouv):
        rows, cols = np.nonzero(iou_same >= thr)     # (gt_idx, pred_idx)
        if rows.size == 0:
            continue
        vals = iou_same[rows, cols]
        order = np.argsort(vals)[::-1]               # IoU 降序
        rows, cols = rows[order], cols[order]
        # 每个 pred 只保留首个(最高 IoU)匹配，然后每个 gt 也只保留首个
        _, c_first = np.unique(cols, return_index=True)
        rows, cols = rows[c_first], cols[c_first]
        _, r_first = np.unique(rows, return_index=True)
        correct[cols[r_first], j] = True
    return correct


def _finish_map(stats_tp: list, stats_conf: list, stats_pred_cls: list,
                stats_gt_cls: list, stats_gt_img: list) -> dict:
    """把逐图累计的 (n,10) tp/conf/类交给开发阶段 NumPy AP 汇总器。

    返回：{map50_95, map50, precision(max-F1), recall(max-F1), ap50_95_per_class, class_ids}
    """
    tp = np.concatenate(stats_tp, 0)
    conf = np.concatenate(stats_conf, 0)
    pcls = np.concatenate(stats_pred_cls, 0)
    tcls = np.concatenate(stats_gt_cls, 0)
    if tp.shape[0] == 0:
        # 一个预测都没有（例如量化崩溃后的退化模型）。仍要返回 n_targets，
        # 否则调用方打印/写 CSV 时会 KeyError——崩溃模型恰恰会走这条路。
        return {"map50_95": 0.0, "map50": 0.0, "precision": 0.0, "recall": 0.0,
                "ap50_95_per_class": np.zeros(80), "class_ids": np.arange(80),
                "n_targets": int(tcls.shape[0]), "n_pred_total": 0}
    p, r, f1, ap, unique_classes = _ap_per_class_numpy(tp, conf, pcls, tcls)
    map50 = float(ap[:, 0].mean())
    map50_95 = float(ap.mean())
    per_class = np.zeros(80)
    per_class[unique_classes] = ap.mean(1)
    return {"map50_95": map50_95, "map50": map50,
            "precision": float(p.mean()), "recall": float(r.mean()),
            "ap50_95_per_class": per_class, "class_ids": unique_classes,
            "n_targets": int(tcls.shape[0])}


class COCOEval:
    """逐图喂预测与真值，结束时输出开发阶段 mAP50-95 / mAP50。

    用法：
        ev = COCOEval()
        for img, (pred_xyxy, conf, cls), (gt_xyxy, gt_cls) in ...:
            ev.push_image(pred_xyxy, conf, cls, gt_xyxy, gt_cls)
        res = ev.finish()
    """

    def __init__(self) -> None:
        self._tp: list[np.ndarray] = []
        self._conf: list[np.ndarray] = []
        self._pred_cls: list[np.ndarray] = []
        self._gt_cls: list[np.ndarray] = []
        self._gt_img: list[np.ndarray] = []

    def push_image(self, pred_xyxy, conf, pred_cls, gt_xyxy, gt_cls, gt_cls_from_json=None) -> None:
        """pred_* 与原图像素 xyxy(N,4)/conf(N,)/cls(N,)；gt 同理(M,4)/(M,)。"""
        pred_xyxy = np.asarray(pred_xyxy, dtype=np.float64).reshape(-1, 4)
        conf = np.asarray(conf, dtype=np.float64).reshape(-1)
        pred_cls = np.asarray(pred_cls).astype(np.int64).reshape(-1)
        gt_xyxy = np.asarray(gt_xyxy, dtype=np.float64).reshape(-1, 4)
        gt_cls = np.asarray(gt_cls).astype(np.int64).reshape(-1)
        n = pred_cls.shape[0]
        if n == 0:
            self._tp.append(np.zeros((0, len(IOUV)), dtype=bool))
        else:
            self._tp.append(match_tp(pred_cls, gt_cls, box_iou(gt_xyxy, pred_xyxy)))
        self._conf.append(conf)
        self._pred_cls.append(pred_cls)
        self._gt_cls.append(gt_cls)
        self._gt_img.append(np.unique(gt_cls))  # 官方口径：每类「至少含一张」计一次

    def finish(self) -> dict:
        return _finish_map(self._tp, self._conf, self._pred_cls, self._gt_cls, self._gt_img)
