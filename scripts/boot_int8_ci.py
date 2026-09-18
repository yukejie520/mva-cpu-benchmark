"""R7 配对 bootstrap：对 500 子集 FP32→INT8 的 mAP Δ 逐图重采样给 95% CI。

原理（为什么能重采样）：coco_eval.match_tp 的匹配逐图独立，eval_dump.py 已把每图
(tp(N,10), conf, pred_cls, gt_cls) 落盘。每次重采样从 500 图**有放回**抽 500 张，
把抽中图的 (tp, conf, cls, gt) 拼起来交给官方 ap_per_class → 重算 mAP。
FP32 与 INT8 用**同一份随机下标** → Δ = mAP_INT8 − mAP_FP32 是配对的。
CI = 各重采样 Δ 的 2.5/97.5 百分位（分位数 bootstrap）。配对差（PAIR_GAPS）在同一重采样
内把两个 scheme 的 Δ 相减，用来判断"配方差"与"格式差"是否超出噪声。

两个正交轴各由一对配对差回答：
    配方差 = naive−sel      （保头 vs 不保头）
    格式差 = qop_sel−sel    （同保头配方，只换 QDQ→QOperator 导出格式）

用法：
    # 只校准 ap_per_class 单次全量耗时（决定 B 取多大合适）
    python scripts/boot_int8_ci.py --dir results/perpix --calib
    # 正式跑（B 取 500~1000），带 QOperator 两个格子
    python scripts/boot_int8_ci.py --dir results/perpix --models n,s \
        --schemes naive,sel,qop,qop_sel --B 1000 --seed 0 \
        --out results/int8_bootstrap_ci.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import numpy as np

# Development aggregation.  The submission-grade full-val path is
# ``evaluate_full_coco_official.py``; these 500-image INT8 deltas use the same
# lightweight per-image representation as ``eval_dump.py`` and do not claim
# to be official COCOeval.
from coco_eval import _ap_per_class_numpy  # noqa: E402

# 归档验证基准：模型 → (fp32 存档 mAP, {scheme: 存档 mAP})（map_subset.csv / map_subset_int8.csv）
# qop / qop_sel 来自 results/map_probe_qop.csv（QOperator 导出格式，同 500 图同管线）
ARCHIVE = {
    "YOLOv8n": {"fp32": 0.395981, "sel": 0.390904, "naive": 0.326641,
                "qop": 0.327815, "qop_sel": 0.387771},
    "YOLOv8s": {"fp32": 0.468496, "sel": 0.466262, "naive": 0.383152,
                "qop": 0.388297, "qop_sel": 0.465675},
    "YOLOv8m": {"fp32": 0.508392, "sel": 0.509606, "naive": 0.421535},
    "YOLOv8l": {"fp32": 0.538881, "sel": 0.529596, "naive": 0.451764},
}

# 配对差（同一份重采样下标内，两个 scheme 的 Δ 相减 → A 相对 B 的精度差）：
#   naive−sel    : 原有，保头 vs 不保头
#   qop−naive    : 新增，**同一 naive 配方只换导出格式**的精度差
#   qop_sel−sel  : 新增，**同一保头配方只换导出格式**的精度差 ← "格式不改变精度"这句的直接检验
# 判读：这三对的 CI 若含 0，说明两边的 ΔmAP 差异在噪声内。
PAIR_GAPS = [("naive", "sel"), ("qop", "naive"), ("qop_sel", "sel")]


def _concat(entries, key):
    """把若干图的数组按行拼起来；空数组安全。"""
    if key == "tp":
        parts = [e["tp"] if e["tp"] is not None else np.zeros((0, 10), dtype=np.uint8)
                 for e in entries]
    else:
        parts = [e[key] for e in entries]
    if not parts:
        return np.zeros(0)
    return np.concatenate(parts, 0)


def mAP_of(entries: list[dict]) -> float:
    """由逐图结果拼全量 → 开发阶段 NumPy mAP50-95。"""
    tp = _concat(entries, "tp").astype(bool)
    conf = _concat(entries, "conf")
    pcls = _concat(entries, "pred_cls").astype(np.int64)
    tcls = _concat(entries, "gt_cls").astype(np.int64)
    if tp.shape[0] == 0:
        return 0.0
    ap = _ap_per_class_numpy(tp, conf, pcls, tcls)[3]
    return float(ap.mean())


def load(label: str, path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)["per_image"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/perpix")
    ap.add_argument("--models", default="n,s,m,l")
    ap.add_argument("--schemes", default="sel,naive", help="要算 CI 的配方（相对 fp32）")
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--calib", action="store_true", help="只计时：每模型单次全量聚合耗时")
    args = ap.parse_args()

    sizes = [s.strip() for s in args.models.split(",")]
    schemes = [s.strip() for s in args.schemes.split(",")]
    d = Path(args.dir)
    data = {}  # model -> {"fp32": entries, scheme: entries}
    for size in sizes:
        model = f"YOLOv8{size}"
        key = {}
        for scheme in ["fp32"] + schemes:
            p = d / f"{model}_{scheme}.pkl"
            if not p.exists():
                print(f"[skip] {model} {scheme}: 缺 {p}")
                continue
            key[scheme] = load(model, p)
        if "fp32" in key:
            data[model] = key

    # 复现校验：重算全量 mAP 与存档对比
    for model, key in data.items():
        fp = mAP_of(key["fp32"])
        print(f"[verify] {model} fp32 重算={fp:.6f} 存档={ARCHIVE[model]['fp32']:.6f} "
              f"Δ={fp - ARCHIVE[model]['fp32']:+.2e}")
        for sc in schemes:
            arch = ARCHIVE.get(model, {}).get(sc)
            if sc in key and arch is not None:
                v = mAP_of(key[sc])
                print(f"         {sc:8s} 重算={v:.6f} 存档={arch:.6f} Δ={v - arch:+.2e}")

    if args.calib:
        t0 = time.perf_counter()
        for model in data:
            mAP_of(data[model]["fp32"])
        dt_ = (time.perf_counter() - t0) / max(1, len(data))
        print(f"[calib] 单次全量聚合 ≈ {dt_:.2f}s/模型 → {args.B} 次重采样 × "
              f"{sum(len(v) for v in data.values())} 次聚合 ≈ "
              f"{dt_ * args.B * sum(len(v) for v in data.values()):.0f}s")
        return

    rng = np.random.default_rng(args.seed)
    n = 500
    # 预拼每个模型的 fp32/配方（免得重采样里反复 join）——直接存 entries 列表。
    # 逐重采样：抽下标 → 用下标列出 entries 再拼。为省时把 per-image 各字段预转成数组栈。
    results = []  # 行: {model, scheme, delta_obs, ci_lo, ci_hi, boot_mean, B}
    gap_rows = []
    t_start = time.perf_counter()
    for model, key in data.items():
        obs_delta = {}
        for sc in schemes:
            if sc not in key:
                continue
            obs_delta[sc] = mAP_of(key[sc]) - mAP_of(key["fp32"])
        if not obs_delta:
            continue
        # 收集每个重采样的各 scheme delta（配对：同一下标）
        boot = {sc: np.zeros(args.B) for sc in obs_delta}
        for b in range(args.B):
            idx = rng.integers(0, n, size=n)
            fp_b = mAP_of([key["fp32"][i] for i in idx])
            for sc in obs_delta:
                boot[sc][b] = mAP_of([key[sc][i] for i in idx]) - fp_b
        for sc in obs_delta:
            d0 = boot[sc]
            results.append({
                "model": model, "scheme": sc,
                "delta_obs": round(obs_delta[sc], 4),
                "boot_mean": round(float(d0.mean()), 4),
                "ci_lo": round(float(np.percentile(d0, 2.5)), 4),
                "ci_hi": round(float(np.percentile(d0, 97.5)), 4),
                "frac_delta_gt0": round(float((d0 > 0).mean()), 3),
                "B": args.B,
            })
            print(f"[ci] {model} {sc:5s} Δ_obs={obs_delta[sc]:+.4f}  "
                  f"95%CI=[{np.percentile(d0, 2.5):+.4f}, {np.percentile(d0, 97.5):+.4f}]")
        # 配对差：A − B（同一下标内相减，仍是配对的）
        for a, b in PAIR_GAPS:
            if a not in obs_delta or b not in obs_delta:
                continue
            gap = boot[a] - boot[b]
            lo, hi = float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))
            crosses0 = lo < 0 < hi
            gap_rows.append({
                "model": model, "scheme": f"{a}-{b}",
                "gap_obs": round(obs_delta[a] - obs_delta[b], 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "frac_gt0": round(float((gap > 0).mean()), 3),
                "ci_crosses_0": bool(crosses0),
            })
            print(f"[gap] {model} {a}−{b} Δ_gap_obs={obs_delta[a] - obs_delta[b]:+.4f} "
                  f"95%CI=[{lo:+.4f}, {hi:+.4f}]  "
                  f"{'跨 0 → 噪声内' if crosses0 else '不跨 0 → 有差异'}")
    print(f"[done] {(time.perf_counter() - t_start):.0f}s  B={args.B}  seed={args.seed}")

    if args.out:
        fields = ["model", "scheme", "delta_obs", "boot_mean", "ci_lo", "ci_hi",
                  "frac_delta_gt0", "B"]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        gfields = ["model", "scheme", "gap_obs", "ci_lo", "ci_hi", "frac_gt0",
                   "ci_crosses_0"]
        with open(Path(args.out).with_name("int8_bootstrap_gap.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=gfields)
            w.writeheader()
            w.writerows(gap_rows)
        print(f"[csv] -> {args.out}")


if __name__ == "__main__":
    main()
