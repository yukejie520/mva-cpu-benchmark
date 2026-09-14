"""批量量化：REGISTRY 全 6 模型 × {dynamic, static} → data/{stem}_{scheme}.onnx。

产出 results/int8_quant.csv：每行记录 体积/成败/报错，供账本 §3。
static 校准集用本地 coco128（64 张，对齐 val 预处理）。
只编排 quantize_int8 里已单测过的函数，本身不引入新逻辑。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from measure import REGISTRY  # noqa: E402
from quantize_int8 import quantize_dynamic_onnx, quantize_static_onnx  # noqa: E402

CALIB = "data/coco128/coco128/images/train2017"
SCHEMES = ["dynamic", "static"]
# static_sel = 双配方里的「selective」：排除输出 Concat + 整个检测头模块(如 model.22)，
# 骨干/颈部仍 INT8 → YOLO 子集 mAP Δ≈-1%（账本 §3 发现①延伸）。RT-DETR 无此配方
# （其 static 数值崩溃，见账本 §3 发现②），仅 CNN 族可用。
SEL_SCHEME = "static_sel"
OUT_CSV = "results/int8_quant.csv"


def scheme_path(cfg: dict, scheme: str) -> Path:
    stem = Path(cfg["onnx"]).stem
    return Path("data") / f"{stem}_{scheme}.onnx"


def _size_mb(p: str | Path) -> float:
    try:
        return round(Path(p).stat().st_size / 1e6, 1)
    except OSError:
        return 0.0


def quantize_one(cfg: dict, scheme: str, calib: str | None = CALIB,
                 calib_n: int = 64) -> dict:
    """对单个 (模型, 方案) 量化，返回结果行（成败都记录，不抛异常）。"""
    fp32 = cfg["onnx"]
    out = scheme_path(cfg, scheme)
    in_mb = _size_mb(fp32)
    t0 = time.perf_counter()
    try:
        if scheme == "dynamic":
            out = quantize_dynamic_onnx(fp32, str(out))
        elif scheme == SEL_SCHEME:
            if cfg.get("family") != "CNN":
                raise ValueError("static_sel 仅 CNN 族（RT-DETR static 数值崩溃，见账本 §3）")
            out = quantize_static_onnx(fp32, str(out), calib, calib_n,
                                       exclude_output_concat=True, exclude_head_module=True)
        else:
            out = quantize_static_onnx(fp32, str(out), calib, calib_n)
        return {"name": cfg["name"], "scheme": scheme, "ok": True, "err": "",
                "in_MB": in_mb, "out_MB": _size_mb(out),
                "seconds": round(time.perf_counter() - t0, 1)}
    except Exception as e:  # noqa: BLE001 —— 记录失败让批次继续
        return {"name": cfg["name"], "scheme": scheme, "ok": False,
                "err": f"{type(e).__name__}: {e}"[:160],
                "in_MB": in_mb, "out_MB": 0.0,
                "seconds": round(time.perf_counter() - t0, 1)}


def run(models: list[str] | None, calib_n: int, out_csv: str,
        schemes: list[str] | None = None) -> list[dict]:
    rows = []
    want = list(schemes) if schemes else list(SCHEMES)
    for cfg in REGISTRY:
        if models and cfg["name"] not in models:
            continue
        if not Path(cfg["onnx"]).exists():
            print(f"[skip] {cfg['name']}: {cfg['onnx']} 不存在")
            continue
        for scheme in want:
            row = quantize_one(cfg, scheme, calib_n=calib_n)
            tag = "ok" if row["ok"] else f"FAIL {row['err'][:80]}"
            print(f"[{cfg['name']} {scheme}] {tag} "
                  f"{row['in_MB']}MB->{row['out_MB']}MB {row['seconds']}s", flush=True)
            rows.append(row)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        fields = ["name", "scheme", "ok", "err", "in_MB", "out_MB", "seconds"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="逗号分隔；默认全 6")
    ap.add_argument("--schemes", default=None,
                    help="逗号分隔 {dynamic,static,static_sel}；默认 dynamic,static")
    ap.add_argument("--calib-n", type=int, default=64)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()
    run(set(args.models.split(",")) if args.models else None, args.calib_n, args.out,
        [s.strip() for s in args.schemes.split(",")] if args.schemes else None)


if __name__ == "__main__":
    main()
