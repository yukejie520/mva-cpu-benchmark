"""S2（2026-09-10）：同一 500 图管线下的七模型精度轴 + bootstrap 95% CI。

要解决的问题（审稿 P0-1）：正文主分析用的是**厂商 COCO 全量 mAP**，而"精度打平"这类判断
建立在跨厂商数字上——两者来自不同评测协议（全量 5000 图 vs 我们取的 500 图子集、不同
预处理/后处理/NMS 口径）。S2 把精度轴换成**我们自己同管线测的 500 图 mAP**，厂商值降级为
附录一致性检查。

数据来源：results/perpix/<model>_fp32.pkl（eval_dump.py 落盘的逐图匹配结果）。
因为 match_tp 的匹配逐图独立，从 500 图有放回重采样再交给官方 ap_per_class 重算，mAP 可精确复现。

三类输出：
1. [verify] 重算全量 mAP 必须与 results/map_subset.csv 存档**逐位一致**（口径自检）。
2. 每模型 bootstrap 95% CI（B 次重采样 500 图 → mAP 的经验分位数区间）。
3. **配对** bootstrap：两个模型共用同一份随机下标（配对重采样）→ ΔmAP 的 CI 与 P(Δ>0)。
   这是"精度打平/谁更高"这类判断的正确统计工具；独立 CI 重叠与否不能回答配对问题。

用法：
    python scripts/map_bootstrap7.py --B 1000 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from boot_int8_ci import mAP_of  # 复用同一聚合口径，避免两套实现漂移
from lae_sweep import MAP_OFFICIAL  # 厂商 mAP 单源

ROOT = Path(__file__).resolve().parents[1]
PERPIX = ROOT / "results" / "perpix"
ARCHIVE_CSV = ROOT / "results" / "map_subset.csv"
OUT_MODELS = ROOT / "results" / "map_bootstrap7.csv"
OUT_PAIRS = ROOT / "results" / "map_paired_bootstrap7.csv"

MODELS = ["YOLO11n", "YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l",
          "RT-DETR-l", "RT-DETR-x"]
# 全部有序对（A − B），42 条。
# 旧版只挑了 5 对"正文关心的"配对，但 §3.4 的 Pareto 边判定规则必须对**所有**对
# 一致适用——只对付得了结论的那几对用规则就是 ad hoc，审稿人一眼看得出。
# 配对数组本身是廉价的（boot 已按模型算好，配对只是做差 + 取分位数），
# 全量输出不增加任何测量成本。
PAIRS = [(a, b) for a in MODELS for b in MODELS if a != b]


def load_entries(label: str) -> list[dict]:
    with open(PERPIX / f"{label}_fp32.pkl", "rb") as f:
        return pickle.load(f)["per_image"]


def draw_indices(rng: np.random.Generator, n: int, b: int) -> np.ndarray:
    """B 份"从 n 图有放回抽 n 张"的下标（配对比较共用同一份）。"""
    return rng.integers(0, n, size=(b, n))


def boot_maps(entries: list[dict], draws: np.ndarray) -> np.ndarray:
    """每份重采样下标算一次 mAP → 长度 B 的经验分布。"""
    return np.array([mAP_of([entries[i] for i in idx]) for idx in draws])


def pair_row(a: str, b_: str, obs: float, lo: float, hi: float,
             frac: float, n_boot: int, seed: int) -> dict:
    """单个配对（A − B）的落盘行。

    这里曾经把 "B" 写了两遍（先模型名、后重采样次数），后者覆盖前者，
    使 CSV 的 B 列丢掉配对模型名、只留下 1000。任何新键都必须与 "B" 区分开。
    """
    return {"A": a, "B": b_, "delta_obs": round(float(obs), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "frac_gt0": round(frac, 3),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": n_boot, "seed": seed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    with open(ARCHIVE_CSV, newline="", encoding="utf-8") as f:
        archive = {r["name"]: float(r["map50_95"]) for r in csv.DictReader(f)}

    entries = {m: load_entries(m) for m in MODELS}

    # ---- 1. 口径自检：重算全量 mAP 必须与存档一致 ----
    print("== 复现校验（重算 vs 存档 map_subset.csv）==")
    for m in MODELS:
        v = mAP_of(entries[m])
        if m in archive:
            flag = "OK" if abs(v - archive[m]) < 1e-9 else "!! 不一致"
            print(f"[verify] {m:11s} 重算={v:.6f} 存档={archive[m]:.6f} Δ={v - archive[m]:+.2e} {flag}")
        else:
            print(f"[verify] {m:11s} 重算={v:.6f} 存档=无（本次新增，同管线同口径）")

    rng = np.random.default_rng(args.seed)
    draws = draw_indices(rng, args.n, args.B)
    boot = {m: boot_maps(entries[m], draws) for m in MODELS}

    # ---- 2. 每模型 CI ----
    print(f"\n== 每模型 bootstrap 95% CI（B={args.B}, seed={args.seed}）==")
    rows = []
    for m in MODELS:
        b = boot[m]
        lo, hi = np.percentile(b, [2.5, 97.5])
        vendor = MAP_OFFICIAL[m]
        rows.append({"model": m, "family": "CNN" if m.startswith("YOLO") else "Transformer",
                     "self_map": round(float(mAP_of(entries[m])), 4),
                     "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                     "boot_sd": round(float(b.std(ddof=1)), 4),
                     "vendor_map": vendor,
                     "delta_self_minus_vendor": round(float(mAP_of(entries[m])) - vendor, 4),
                     "n_imgs": args.n, "B": args.B, "seed": args.seed})
        print(f"[ci] {m:11s} self={mAP_of(entries[m]):.4f}  CI=[{lo:.4f}, {hi:.4f}]  "
              f"vendor={vendor:.3f}  Δ={mAP_of(entries[m]) - vendor:+.4f}")

    # ---- 3. 配对 Δ（共用下标 → 真正的配对重采样）----
    print("\n== 配对 bootstrap：A − B（共用同一份重采样下标）==")
    prows = []
    for a, b_ in PAIRS:
        d = boot[a] - boot[b_]
        obs = mAP_of(entries[a]) - mAP_of(entries[b_])
        lo, hi = np.percentile(d, [2.5, 97.5])
        frac = float((d > 0).mean())
        prows.append(pair_row(a, b_, float(obs), float(lo), float(hi),
                              frac, args.B, args.seed))
        verdict = "CI 含 0 → 精度无统计差别" if (lo <= 0 <= hi) else "CI 不含 0 → 有差别"
        print(f"[pair] {a:11s} − {b_:11s} Δ={obs:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  "
              f"P(Δ>0)={frac:.3f}  {verdict}")

    with open(OUT_MODELS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_PAIRS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(prows[0].keys()))
        w.writeheader()
        w.writerows(prows)
    print(f"\n[csv] -> {OUT_MODELS.name}, {OUT_PAIRS.name}")


if __name__ == "__main__":
    main()
