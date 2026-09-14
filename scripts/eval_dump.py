"""逐图匹配结果落盘：供 R7 配对 bootstrap 重采样（500 子集 FP32 / selective-INT8）。

为什么需要（R7 补测，2026-09-09）：map_subset*.csv 只存聚合 mAP。mAP 是跨图累计的
PR 曲线，不是逐图均值，无法从聚合倒推重采样 → 须把每张图的**匹配结果**存档。
关键事实：coco_eval.match_tp 的贪心匹配是逐图独立的（本图预测×本图真值），所以
(tp, conf, cls, gt_cls) 每图一份，重采样任意图集都能精确重算 mAP，不必重跑前向。

口径与 eval_map 完全一致（同 _Subset/同 decode/同 conf=0.001/iou=0.7/max_det=300），
保证重跑聚合 mAP 与存档 map_subset.csv / map_subset_int8.csv 精确吻合（复现校验）。

用法：
    python scripts/eval_dump.py --onnx data/yolov8l.onnx --label YOLOv8l_fp32 \
        --anns data/coco_val500/instances_val500.json --img-root data/coco_val500/images \
        --out results/perpix/YOLOv8l_fp32.pkl
    # 冒烟：加 --n 20 只评前 20 图
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from coco_eval import COCOEval, box_iou, match_tp  # noqa: E402
from eval_common import build_session  # noqa: E402
from eval_map import (MAX_DET, _Subset, decode_rtdetr, decode_yolo,  # noqa: E402
                      preprocess_meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--label", required=True, help="落盘 pickle 的名字，如 YOLOv8l_fp32")
    ap.add_argument("--kind", choices=["yolo", "rtdetr"], required=True)
    ap.add_argument("--anns", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out", required=True, help="输出 .pkl 路径")
    ap.add_argument("--n", type=int, default=0, help="只评前 n 图（冒烟）")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    subset = _Subset(args.anns)
    sess = build_session(args.onnx)
    ev = COCOEval()
    inp_name = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    n_imgs = len(subset.images) if not args.n else min(args.n, len(subset.images))
    t0 = time.perf_counter()
    per_image: list[dict] = []
    for im in subset.images[:n_imgs]:
        img_path = Path(args.img_root) / im["file_name"]
        if not img_path.exists():
            raise FileNotFoundError(f"缺图 {img_path}")
        x, meta = preprocess_meta(str(img_path))
        raw = sess.run(outs, {inp_name: x})[0]
        if args.kind == "yolo":
            boxes, confs, clss = decode_yolo(raw, meta, args.conf, args.iou)
        else:
            # S2（2026-09-10）：RT-DETR 支补上，口径与 eval_map 完全一致（免 NMS、conf>=0.001）。
            boxes, confs, clss = decode_rtdetr(raw, meta, args.conf)
        gt_b, gt_c = subset.gt[im["id"]]
        ev.push_image(boxes, confs, clss, gt_b, gt_c)  # 顺带算聚合，供复现校验
        # 逐图匹配结果落盘：tp(N,10) 用 uint8 省空间
        if clss.shape[0]:
            tp = match_tp(clss, gt_c, box_iou(gt_b, boxes))
        else:
            tp = None
        per_image.append({
            "file": im["file_name"],
            "tp": tp,                      # (N,10) bool 或 None(无预测)
            "conf": confs.astype("float64"),
            "pred_cls": clss.astype("int64"),
            "gt_cls": gt_c.astype("int64"),
        })
    res = ev.finish()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"label": args.label, "images": n_imgs, "per_image": per_image}, f)
    print(f"[{args.label}] n_imgs={n_imgs} n_targets={res['n_targets']} "
          f"mAP50-95={res['map50_95']:.6f}  ({(time.perf_counter() - t0):.1f}s) -> {args.out}")


if __name__ == "__main__":
    main()
