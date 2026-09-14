"""交错测量（2026-09-10）：把"逐变体顺序跑"换成"每轮轮流跑一遍"。

为什么要交错（上一轮实测踩到的坑）：
    同一台机器、同一会话配置，YOLOv8n FP32 先测到 29.83 ms、后测到 31.78 ms —— 相差 6.5%。
    原因是顺序测量把"变体差异"和"时间漂移（升温/降频/调度）"混在一起：排在后面的变体
    系统性地处在更热、频率更低的状态。当前实测有效频率只有 1.4-1.9 GHz（标称基频 2.2 GHz），
    说明这台机器确实在降频，漂移不是理论风险。

交错怎么解决：
    每一轮里让所有变体各测一遍，轮内顺序逐轮旋转（round-robin），于是每个变体在每轮都经历
    相近的热状态。关键统计量是**轮内配对比值** r_i = t_baseline(第 i 轮) / t_variant(第 i 轮)，
    它在同一轮内做除法，天然消掉了那一轮的公共频率/温度水平。
    报告 R 个配对比值的中位数与范围，而不是两个各自跨轮的中位数的商。

频率记录：每轮每变体前后各采一次 Windows 计数器
(Processor Information / % Processor Performance)，
乘以 MaxClockSpeed 得有效 MHz（>100% 即睿频）。

用法：
    python scripts/measure_interleaved.py --rounds 5 --out results/interleaved_int8_yolov8n.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_common import IMGSZ, build_session, list_images, measure_forward, preprocess  # noqa: E402
from platform_telemetry import cpu_frequency_mhz, cpu_temperature_c  # noqa: E402
from protocol_io import image_paths_from_manifest, load_int8_triplet  # noqa: E402

CALIB = ROOT / "data" / "coco128" / "coco128" / "images" / "train2017"
DATA = ROOT / "data"

# 变体：标签 -> onnx 文件后缀。baseline 必须第一个（配对比值以它为分母）。
VARIANTS_SUFFIX = [
    ("FP32", ""),
    ("QDQ_naive", "_static"),
    ("QDQ_sel", "_static_sel"),
    ("QOp_naive", "_static_qop"),
    ("QOp_sel", "_static_qop_sel"),
]
MODEL = "yolov8n"


def variants_for(model: str, artifact_manifest: str | None = None) -> list[tuple[str, Path]]:
    """Return variants, using an explicit verified map when S2 requests it."""
    if artifact_manifest:
        artifacts = load_int8_triplet(artifact_manifest, ROOT, model)
        return [("FP32", artifacts["FP32"]),
                ("QDQ", artifacts["QDQ"]),
                ("QOperator", artifacts["QOperator"])]
    return [(tag, DATA / f"{model}{suf}.onnx") for tag, suf in VARIANTS_SUFFIX]


# 注意：dynamic（YOLOv8n 约 238 ms，慢 8 倍）不纳入交错——它每题都要占掉一个变体
# 10 倍的时间，且在正文里的角色（权重-only 量化的下界）不依赖本轮这几毫秒的精度。

_PS = (
    r"$s=1..3|%{(Get-Counter '\Processor Information(_Total)\% Processor Performance')"
    r".CounterSamples.CookedValue};"
    r"$m=(Get-CimInstance Win32_Processor).MaxClockSpeed;"
    r"'{0} {1}' -f $m,($s|Measure-Object -Average).Average"
)


def parse_mhz(stdout: str) -> float:
    """把 PowerShell 的 'max avg' 两数解析成有效 MHz。纯函数，便于测试。"""
    parts = stdout.split()
    if len(parts) < 2:
        raise ValueError(f"频率读数格式不对: {stdout!r}")
    max_mhz, avg_pct = float(parts[0]), float(parts[1])
    return max_mhz * avg_pct / 100.0


def cpu_mhz() -> float:
    """当前有效 CPU 频率（MHz）；取不到就返回 nan，绝不编造。"""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", _PS],
                             capture_output=True, text=True, timeout=30)
        return parse_mhz(out.stdout.strip())
    except Exception:
        value = cpu_frequency_mhz()
        return value if value is not None else float("nan")


def rotation_orders(n: int, rounds: int) -> list[list[int]]:
    """R 轮、每轮 n 个变体的轮内顺序：第 r 轮从第 r 个开始循环右移。

    这样当 rounds >= n 时，每个变体在"第几个被跑"的每个位置上出现次数尽量相等，
    把"排在前面/后面"这个位置效应平摊掉。
    """
    return [[(r + i) % n for i in range(n)] for r in range(rounds)]


def round_matched_ratios(base: list[float], var: list[float]) -> list[float]:
    """逐轮配对比值 base_i / var_i（同一轮内做除法 → 消掉该轮的公共热状态）。"""
    if len(base) != len(var):
        raise ValueError(f"轮数不齐：base={len(base)} var={len(var)}")
    return [b / v for b, v in zip(base, var)]


def median_of_medians(per_round: list[float]) -> float:
    return st.median(per_round)


def summarize(base_rounds: list[float], var_rounds: list[float]) -> dict:
    """产出一行汇总：中位延迟（轮间中位数）与轮内配对比值的中位/最小/最大。"""
    r = sorted(round_matched_ratios(base_rounds, var_rounds))
    round_ratios = round_matched_ratios(base_rounds, var_rounds)
    return {
        "median_ms": round(median_of_medians(var_rounds), 4),
        "ratio_median": round(st.median(r), 4),
        "ratio_min": round(r[0], 4),
        "ratio_max": round(r[-1], 4),
        "ratio_all_gt1": bool(r[0] > 1.0),
        "ratio_all_lt1": bool(r[-1] < 1.0),
        "speed_direction": ("faster" if r[0] > 1.0 else
                             "slower" if r[-1] < 1.0 else "not_separated"),
        "ratio_rounds": "|".join(f"{x:.4f}" for x in round_ratios),
        "round_medians_ms": "|".join(f"{x:.3f}" for x in var_rounds),
    }


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--imgs", default=str(CALIB), help="输入图像目录")
    ap.add_argument("--manifest", default=None,
                    help="使用记录的图像清单，而不是重新排序取前 n 张")
    ap.add_argument("--artifact-manifest", default=None,
                    help="使用已核验的 FP32/QDQ/QOperator 工件清单")
    ap.add_argument("--platform", default="unspecified",
                    help="写入 metadata 的平台标签")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    default_name = (f"interleaved_cross_isa_{args.model}.csv"
                    if args.artifact_manifest else f"interleaved_int8_{args.model}.csv")
    out_path = Path(args.out or str(ROOT / "results" / default_name))

    variants = variants_for(args.model, args.artifact_manifest)
    image_paths = (image_paths_from_manifest(args.manifest, args.imgs, args.n_imgs)
                   if args.manifest else list_images(str(args.imgs), args.n_imgs))
    inputs = [preprocess(str(p), IMGSZ) for p in image_paths]
    labels = [v[0] for v in variants]
    sess = {tag: build_session(str(path), threads=args.threads) for tag, path in variants}

    print(f"[in] {len(inputs)} 张图 {Path(args.imgs).name}；线程={args.threads}；预热={args.warmup}；"
          f"{args.rounds} 轮 × {args.reps} 次；变体={labels}")
    print(f"[freq] 起始有效频率 {cpu_mhz():.0f} MHz\n")

    run_started = _iso_now()
    raw_rows, sample_rows = [], []
    per_round, freq_rows, temp_rows = {t: [] for t in labels}, [], []
    round_metadata = []
    for r, order in enumerate(rotation_orders(len(variants), args.rounds)):
        round_started = _iso_now()
        round_metadata.append({"round": r + 1, "order": [labels[i] for i in order],
                               "started_utc": round_started})
        print(f"---- 第 {r + 1}/{args.rounds} 轮  轮内顺序: {[labels[i] for i in order]} ----",
              flush=True)
        for i in order:
            tag, _path = variants[i]
            variant_started = _iso_now()
            before = cpu_mhz()
            temp_before = cpu_temperature_c()
            ts = measure_forward(sess[tag], inputs, warmup=args.warmup, reps=args.reps)
            after = cpu_mhz()
            temp_after = cpu_temperature_c()
            variant_finished = _iso_now()
            med = st.median(ts)
            per_round[tag].append(med)
            raw_rows.append({"round": r + 1, "order": order.index(i), "variant": tag,
                             "file": str(_path),
                             "median_ms": round(med, 4), "n_samples": len(ts),
                             "freq_before_mhz": round(before, 1) if before is not None else "",
                             "freq_after_mhz": round(after, 1) if after is not None else "",
                             "temp_before_c": round(temp_before, 2) if temp_before is not None else "",
                             "temp_after_c": round(temp_after, 2) if temp_after is not None else "",
                             "started_utc": variant_started, "finished_utc": variant_finished})
            for sample_index, value in enumerate(ts):
                sample_rows.append({
                    "round": r + 1, "order": order.index(i), "variant": tag,
                    "file": str(_path), "image_index": sample_index // args.reps,
                    "rep": sample_index % args.reps + 1,
                    "latency_ms": round(value, 4),
                    "freq_before_mhz": round(before, 1) if before is not None else "",
                    "freq_after_mhz": round(after, 1) if after is not None else "",
                    "temp_before_c": round(temp_before, 2) if temp_before is not None else "",
                    "temp_after_c": round(temp_after, 2) if temp_after is not None else "",
                })
            if before is not None and after is not None:
                freq_rows.append((before + after) / 2)
            if temp_before is not None and temp_after is not None:
                temp_rows.append((temp_before + temp_after) / 2)
            print(f"   {tag:10s} {med:8.3f} ms   freq {before:.0f}->{after:.0f} MHz", flush=True)
        print()
        round_metadata[-1]["finished_utc"] = _iso_now()

    base = per_round["FP32"]
    print("===== 轮内配对比值（分母 = 同轮 FP32）=====")
    print(f"  {'变体':10s} {'中位ms':>9s} {'比值中位':>9s} {'最小':>7s} {'最大':>7s}  "
          f"{'每轮全部 >1':>10s}   轮内中位数")
    summary = []
    for tag, _ in variants:
        s = summarize(base, per_round[tag])
        s.update({"variant": tag, "rounds": args.rounds})
        summary.append(s)
        print(f"  {tag:10s} {s['median_ms']:9.3f} {s['ratio_median']:9.3f} "
              f"{s['ratio_min']:7.3f} {s['ratio_max']:7.3f}  "
              f"{'是' if s['ratio_all_gt1'] else '否':>10s}   {s['round_medians_ms']}")

    valid = [f for f in freq_rows if f == f]  # 滤掉 nan
    if valid:
        print(f"\n[freq] 全程有效频率 中位={st.median(valid):.0f} MHz  "
              f"最低={min(valid):.0f}  最高={max(valid):.0f}  （标称基频 2200）")
    if temp_rows:
        print(f"[temp] 中位={st.median(temp_rows):.1f} C  最低={min(temp_rows):.1f} 最高={max(temp_rows):.1f}")

    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)
    sout = out.with_name(out.stem + "_summary.csv")
    with open(sout, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    sample_out = out.with_name(out.stem + "_samples.csv")
    with open(sample_out, "w", newline="", encoding="utf-8") as f:
        fields = list(sample_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sample_rows)
    metadata_out = out.with_name(out.stem + "_metadata.json")
    metadata = {
        "platform": args.platform,
        "model": args.model,
        "artifact_manifest": str(args.artifact_manifest) if args.artifact_manifest else None,
        "image_dir": str(args.imgs),
        "image_manifest": str(args.manifest) if args.manifest else None,
        "images": [str(p) for p in image_paths],
        "variants": [{"label": tag, "file": str(path)} for tag, path in variants],
        "threads": args.threads, "batch_size": 1, "imgsz": IMGSZ,
        "warmup": args.warmup, "reps": args.reps, "rounds": args.rounds,
        "started_utc": run_started, "finished_utc": _iso_now(),
        "rounds_detail": round_metadata,
    }
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print(f"\n[csv] -> {out.name}, {sout.name}, {sample_out.name}")
    print(f"[metadata] -> {metadata_out.name}")


if __name__ == "__main__":
    main()
