"""本地开发阶段 mAP 评估驱动：ORT 前向（FP32/INT8 通用）+ NumPy AP。

用途：账本 §3 的 INT8 mAPΔ。对同一批子集图分别跑 FP32 与 INT8，
两者用同一评估代码/同一匹配口径 → Δ 就是"量化掉多少精度"。

关键坐标事实（已与官方 torch 输出逐框对齐，2026-09-06）：
- YOLOv8 ONNX：输出 (1,84,8400)，前4=cxcywh 已是 640 画布像素、后80已含 sigmoid
  → yolov8_nms 得画布 xyxy → to_orig。
- RT-DETR ONNX：输出 (1,300,6)，前4=[cx,cy,w,h] **对 640 画布归一化到 0~1**，
  第5=conf，第6=cls。→ ×640 得画布 cxcywh → 转 xyxy → to_orig（免 NMS）。
- 匹配在**原图像素**进行（IoU 匹配与坐标系无关，只需双方同空间）。

口径：conf>=0.001（COCO AP 惯例）；YOLO NMS iou=0.7、max_det=300（对齐 ultralytics val）。
此脚本剔除 iscrowd，且不执行 COCOeval 的 maxDets/面积分层处理，因此只能用于开发阶段
相对比较；投稿用绝对精度请运行 ``evaluate_full_coco_official.py``。

用法：
    python scripts/eval_map.py --onnx data/yolov8n.onnx --name YOLOv8n --kind yolo \
        --anns data/coco_val500/instances_val500.json --img-root data/coco_val500/images \
        --out results/map_subset.csv [--n 50] [--conf 0.001]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from coco_eval import COCOEval, category_to_idx, xywh_to_xyxy  # noqa: E402
from eval_common import IMGSZ, build_session, letterbox  # noqa: E402
from yolo_post import yolov8_nms  # noqa: E402

MAX_DET = 300


def preprocess_meta(path: str, imgsz: int = IMGSZ) -> tuple[np.ndarray, dict]:
    """letterbox 预处理（调用 eval_common.letterbox 保持单一来源）+ 反映射元信息。

    元信息：r(原图->画布缩放)、pl/pt(左/上 pad)、orig_hw。返回 (1,3,H,W) 输入, meta。
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read {path}")
    h, w = img.shape[:2]
    lb = letterbox(img, imgsz)
    r = min(imgsz / h, imgsz / w)
    nw, nh = round(w * r), round(h * r)
    pl, pt = int(round((imgsz - nw) / 2 - 0.1)), int(round((imgsz - nh) / 2 - 0.1))
    arr = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(arr.transpose(2, 0, 1), 0), {"r": r, "pl": pl, "pt": pt, "hw": (h, w)}


def to_orig(xyxy_canvas: np.ndarray, meta: dict) -> np.ndarray:
    """画布 xyxy -> 原图像素 xyxy（裁剪越界）。"""
    b = np.asarray(xyxy_canvas, dtype=np.float64).copy()
    r, pl, pt, (h, w) = meta["r"], meta["pl"], meta["pt"], meta["hw"]
    b[:, [0, 2]] = np.clip((b[:, [0, 2]] - pl) / r, 0, w - 1)
    b[:, [1, 3]] = np.clip((b[:, [1, 3]] - pt) / r, 0, h - 1)
    return b


def decode_yolo(raw: np.ndarray, meta: dict, conf: float, iou: float) -> tuple:
    boxes, confs, clss = yolov8_nms(raw, conf_thres=conf, iou_thres=iou, max_det=MAX_DET)
    if boxes.shape[0]:
        boxes = to_orig(boxes, meta)
    return boxes, confs, clss


def decode_rtdetr(raw: np.ndarray, meta: dict, conf: float) -> tuple:
    """(1,300,6) 前4 为对 640 画布归一化的 cxcywh。返回 原图 xyxy/conf/cls(过滤后)。"""
    o = np.asarray(raw[0], dtype=np.float64)
    confs = o[:, 4]
    keep = confs >= conf
    if not keep.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
    cxcywh = o[keep, :4] * IMGSZ
    x1 = cxcywh[:, 0] - cxcywh[:, 2] / 2
    y1 = cxcywh[:, 1] - cxcywh[:, 3] / 2
    canvas = np.column_stack([x1, y1, x1 + cxcywh[:, 2], y1 + cxcywh[:, 3]])
    return to_orig(canvas, meta), confs[keep], o[keep, 5].astype(np.int64)


class _Subset:
    """从子集 json 读图清单与逐图真值（原图像素 xyxy + 0..79 类）。"""

    def __init__(self, anns_json: str):
        coco = json.loads(Path(anns_json).read_text(encoding="utf-8"))
        self.images = sorted(coco["images"], key=lambda d: d["file_name"])
        ann_by_im: dict[int, list] = {}
        for a in coco["annotations"]:
            if a.get("iscrowd"):
                continue
            ann_by_im.setdefault(a["image_id"], []).append(a)
        self.gt: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for im in self.images:
            anns = ann_by_im.get(im["id"], [])
            if not anns:
                self.gt[im["id"]] = (np.zeros((0, 4)), np.zeros(0, dtype=np.int64))
                continue
            boxes = xywh_to_xyxy(np.array([a["bbox"] for a in anns]))
            cats = np.array([category_to_idx(a["category_id"]) for a in anns], dtype=np.int64)
            self.gt[im["id"]] = (boxes, cats)


def run(args) -> dict:
    subset = _Subset(args.anns)
    sess = build_session(args.onnx, threads=args.threads)
    ev = COCOEval()
    t0 = time.perf_counter()
    n_imgs = len(subset.images) if not args.n else min(args.n, len(subset.images))
    n_pred_total = 0
    inp_name = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    for im in subset.images[:n_imgs]:
        img_path = Path(args.img_root) / im["file_name"]
        if not img_path.exists():
            raise FileNotFoundError(f"缺图 {img_path}；先跑 prepare_val_subset.py")
        x, meta = preprocess_meta(str(img_path))
        raw = sess.run(outs, {inp_name: x})[0]
        if args.kind == "yolo":
            boxes, confs, clss = decode_yolo(raw, meta, args.conf, args.iou)
        else:
            boxes, confs, clss = decode_rtdetr(raw, meta, args.conf)
        gt_b, gt_c = subset.gt[im["id"]]
        ev.push_image(boxes, confs, clss, gt_b, gt_c)
        n_pred_total += boxes.shape[0]
    res = ev.finish()
    res.update({"name": args.name, "kind": args.kind, "images": n_imgs,
                "n_pred": n_pred_total,
                "seconds": round(time.perf_counter() - t0, 1)})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--kind", choices=["yolo", "rtdetr"], required=True)
    ap.add_argument("--anns", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out", default=None, help="append 一行到 CSV")
    ap.add_argument("--n", type=int, default=0, help="只评前 n 图（调试/冒烟）")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    res = run(args)
    print(f"[{res['name']}] n_imgs={res['images']} n_targets={res['n_targets']} "
          f"mAP50-95={res['map50_95']:.3f}  mAP50={res['map50']:.3f}  "
          f"P={res['precision']:.3f} R={res['recall']:.3f}  ({res['seconds']}s)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fields = ["name", "kind", "images", "n_targets", "n_pred",
                  "map50_95", "map50", "precision", "recall", "seconds"]
        new = not Path(args.out).exists()
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow({k: res[k] for k in fields})


if __name__ == "__main__":
    main()
