"""LAE 指标：α×β 秩稳定性扫描 + 朴素效率指标对照 + Pareto 有效性（W5 分析）。

数据源（单源真值纪律）：
- 延迟 / 参数量 / GFLOPs：读 `results/latency_canonical7.csv`（2026-09-07 单窗口规范表，
  账本 §2 唯一真源。e2e_median_ms 已是 3 轮中位数之中位数）。
- accuracy reference: the vendor/model-card table below remains the compatibility
  baseline for historical LAE/bootstrap scripts. A locally re-evaluated table must
  not be substituted here until it has passed the standard COCOeval audit.

口径（账本 §5 拍板）：
- LAE = mAP / (Latency_ms^α × Params_M^β)，默认 α=0.5、β=0.3（固定指数，不拟合）。
- 秩稳定性：α×β 网格内全排序与默认排序逐点比较 → 身份命中率 / 每模型位移 / top-1 稳定率。
  红线：绝不在参与评分的 7 模型上拟合 α/β —— 本脚本只是"扫固定网格看排序变不变"。
- 朴素对照（自定义，ODEI 精确公式不可得，只作背景引用）：
    E_FLOPs = mAP / GFLOPs   （硬件无关的算力效率，ODEI 一类）
    E_Lat   = mAP / Latency_ms（纯延迟效率 ≡ LAE(α=1,β=0)，网格角点）
    E_Params= mAP / Params_M （纯参数量效率 ≡ LAE(α=0,β=1)，网格角点）
- Pareto 有效性：三目标支配 = A 支配 B 当 map_A≥map_B ∧ L_A≤L_B ∧ P_A≤P_B（至少一个严格）。

用法：
    python scripts/lae_sweep.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Historical external reference values.  Keep this name stable because
# map_bootstrap7.py and the archived INT8 reports import it.  In particular,
# do not silently replace these with the current local NumPy evaluator output.
MAP_OFFICIAL = {
    "YOLO11n": 0.395,
    "YOLOv8n": 0.373,
    "YOLOv8s": 0.449,
    "YOLOv8m": 0.502,
    "YOLOv8l": 0.529,
    "RT-DETR-l": 0.530,
    "RT-DETR-x": 0.548,
}


def load_full_map(path: Path | str) -> dict[str, float]:
    """Load a *validated* local full-val table explicitly supplied by a caller."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    values = {r["name"]: float(r["map50_95"]) for r in rows}
    if len(values) != 7 or any(int(r["images"]) != 5000 for r in rows):
        raise ValueError(f"{path} must contain seven 5,000-image rows")
    return values
DEFAULT_ALPHA, DEFAULT_BETA = 0.5, 0.3
CANON_CSV = Path(__file__).resolve().parents[1] / "results" / "latency_canonical7.csv"


def load_canonical(path: Path | str = CANON_CSV) -> list[dict]:
    """Read the canonical latency table and join the unified full-val mAP."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["name"] not in MAP_OFFICIAL:
                continue
            rows.append({
                "name": r["name"],
                "map": MAP_OFFICIAL[r["name"]],
                "lat_ms": float(r["e2e_median_ms"]),
                "params_m": float(r["params_M"]),
                "gflops": float(r["gflops"]),
            })
    if len(rows) != len(MAP_OFFICIAL):
        raise ValueError(f"{path} 只含 {len(rows)}/{len(MAP_OFFICIAL)} 个带官方 mAP 的模型")
    return rows


def lae_score(map_: float, lat_ms: float, params_m: float,
              alpha: float, beta: float) -> float:
    """LAE = mAP / (Latency^α × Params^β)。mAP 越大越好、L/P 越小越好 → 指数取正。"""
    return map_ / (lat_ms ** alpha * params_m ** beta)


def ranking(rows: list[dict], alpha: float, beta: float) -> list[str]:
    """按 LAE 得分降序排出模型名。得分相同时按名字字典序（确定性，实际不会并列）。"""
    return sorted((r["name"] for r in rows),
                  key=lambda n: (-lae_score(*_cols(rows, n), alpha, beta), n))


def _cols(rows: list[dict], name: str) -> tuple[float, float, float]:
    r = next(r for r in rows if r["name"] == name)
    return r["map"], r["lat_ms"], r["params_m"]


def rank_positions(ordered: list[str]) -> dict[str, int]:
    """排名列表 → {模型名: 名次(1=第一)}。"""
    return {n: i + 1 for i, n in enumerate(ordered)}


def spearman(pos_a: dict[str, int], pos_b: dict[str, int]) -> float:
    """两排名向量的 Spearman 秩相关（对秩做 Pearson）。名字集合须一致。"""
    names = sorted(pos_a)
    if names != sorted(pos_b):
        raise ValueError("两排名覆盖的模型集合不一致")
    ra = np.array([pos_a[n] for n in names], dtype=float)
    rb = np.array([pos_b[n] for n in names], dtype=float)
    if ra.size == 0 or np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra - ra.mean(), rb - rb.mean())[0, 1])


def pareto_dominated(rows: list[dict]) -> set[str]:
    """三目标严格支配下的被支配模型：存在另一模型在 map 不更低、lat/params 不更高且至少一项更优。"""
    dominated: set[str] = set()
    for a in rows:
        for b in rows:
            if a["name"] == b["name"]:
                continue
            b_no_worse = (b["map"] >= a["map"] and b["lat_ms"] <= a["lat_ms"]
                          and b["params_m"] <= a["params_m"])
            b_strict = (b["map"] > a["map"] or b["lat_ms"] < a["lat_ms"]
                        or b["params_m"] < a["params_m"])
            if b_no_worse and b_strict:
                dominated.add(a["name"])
                break
    return dominated


def grid_stability(rows: list[dict], alpha_vals: list[float],
                   beta_vals: list[float]) -> dict:
    """扫 α×β 网格：每点算全排序，对比默认排序 → 返回统计 + 逐点明细。"""
    default = rank_positions(ranking(rows, DEFAULT_ALPHA, DEFAULT_BETA))
    default_names = [n for n, _ in sorted(default.items(), key=lambda kv: kv[1])]
    cells = []
    for a in alpha_vals:
        for b in beta_vals:
            pos = rank_positions(ranking(rows, a, b))
            cells.append({
                "alpha": a, "beta": b,
                "identical": pos == default,
                "rho": spearman(pos, default),
                "order": ">".join(n for n, _ in sorted(pos.items(), key=lambda kv: kv[1])),
            })
    identity = [c for c in cells if c["identical"]]
    # 每模型在全网格可达名次范围
    pos_ranges: dict[str, list[int]] = {n: [] for n in default}
    for a, b in ((c["alpha"], c["beta"]) for c in cells):
        pos = rank_positions(ranking(rows, a, b))
        for n in default:
            pos_ranges[n].append(pos[n])
    range_of = {n: (min(v), max(v)) for n, v in pos_ranges.items()}
    top1 = {c["order"].split(">")[0] for c in cells}
    return {
        "n_cells": len(cells),
        "n_identity": len(identity),
        "identity_rate": len(identity) / len(cells),
        "min_rho": min(c["rho"] for c in cells),
        "range_of": range_of,
        "top1_names": sorted(top1),
        "top1_stable": len(top1) == 1,
        "cells": cells,
    }


def naive_baselines(rows: list[dict]) -> dict[str, list[str]]:
    """自定义朴素指标排序（含 =LAE(α=1,β=0) 与 =LAE(α=0,β=1) 的注释关系）。"""
    def order(key):
        return [r["name"] for r in
                sorted(rows, key=lambda r: -r["map"] / key(r))]
    return {
        "E_FLOPs": order(lambda r: r["gflops"]),   # 硬件无关算力效率（ODEI 类）
        "E_Lat": order(lambda r: r["lat_ms"]),      # ≡ LAE(α=1, β=0)（网格角点）
        "E_Params": order(lambda r: r["params_m"]),  # ≡ LAE(α=0, β=1)（网格角点）
    }


def rank_vs_default(pos: dict[str, int], default: dict[str, int]) -> float:
    return spearman(pos, default)


def select_best(rows: list[dict], min_map: float, max_lat: float,
                max_params: float, metric: str = "LAE") -> str | None:
    """选型协议查询：在满足预算（mAP≥min_map、lat≤max_lat、params≤max_params）的模型里
    按 metric 挑 LAE 最高者。metric 仅支持 'LAE'（用默认指数）。无满足者返回 None。"""
    cand = [r for r in rows
            if r["map"] >= min_map and r["lat_ms"] <= max_lat
            and r["params_m"] <= max_params]
    if not cand:
        return None
    if metric == "LAE":
        return max(cand, key=lambda r: lae_score(r["map"], r["lat_ms"],
                                                 r["params_m"], DEFAULT_ALPHA,
                                                 DEFAULT_BETA))["name"]
    raise ValueError(f"不支持的 metric: {metric}")


def _fmt_rank(order: list[str]) -> str:
    return " > ".join(order)


def main() -> None:
    rows = load_canonical()
    # 计划网格：α∈[0.2,0.8] step0.05、β∈[0.1,0.5] step0.05（13×9=117 点）
    alphas = np.round(np.arange(0.2, 0.801, 0.05), 2).tolist()
    betas = np.round(np.arange(0.1, 0.501, 0.05), 2).tolist()

    default_order = ranking(rows, DEFAULT_ALPHA, DEFAULT_BETA)
    default_pos = rank_positions(default_order)
    print("== LAE 默认 (α=0.5, β=0.3) ==")
    print("rank: " + _fmt_rank(default_order))

    stab = grid_stability(rows, alphas, betas)
    print(f"\n== α×β 网格 {stab['n_cells']} 点 (α∈[0.2,0.8], β∈[0.1,0.5]) ==")
    print(f"身份命中（排序与默认逐点全同）: {stab['n_identity']}/{stab['n_cells']}"
          f" = {stab['identity_rate']:.1%}")
    print(f"每模型可达名次范围: "
          + ", ".join(f"{n} {lo}-{hi}" for n, (lo, hi) in stab["range_of"].items()))
    print(f"top-1 稳定: {'是 (' + stab['top1_names'][0] + ')' if stab['top1_stable'] else '否'}")
    print(f"min Spearman(网格点, 默认) = {stab['min_rho']:.4f}")

    dom = pareto_dominated(rows)
    frontier = [r["name"] for r in rows if r["name"] not in dom]
    print(f"\n== Pareto（三目标 map↑/lat↓/params↓）==")
    print(f"被支配: {sorted(dom) or '无'};  前沿: {frontier}")
    print(f"默认 top-1 在解集内: {default_order[0] not in dom}")

    baselines = naive_baselines(rows)
    print("\n== 朴素指标对照（Spearman vs LAE 默认；命中=排序与默认逐点全同）==")
    for name, order in baselines.items():
        rho = rank_vs_default(rank_positions(order), default_pos)
        print(f"{name:9s} rho={rho:.3f}  全同={order == default_order}  rank: {_fmt_rank(order)}")

    # 效率余量失真（论文"为什么需要实测延迟指标"素材）：FLOPs 效率 vs 实测延迟效率的余量比
    print("\n== 效率余量对照（FLOPs 效率 vs 实测延迟效率；余量=高排名者/低排名者）==")
    for hi, lo in [("YOLOv8n", "YOLOv8l"), ("YOLOv8l", "RT-DETR-l"),
                   ("RT-DETR-l", "RT-DETR-x")]:
        rh, rl = next(r for r in rows if r["name"] == hi), next(r for r in rows if r["name"] == lo)
        m = rh["map"] / rl["map"]
        f_ratio = (rh["map"] / rh["gflops"]) / (rl["map"] / rl["gflops"])
        t_ratio = (rh["map"] / rh["lat_ms"]) / (rl["map"] / rl["lat_ms"])
        p_ratio = (rh["map"] / rh["params_m"]) / (rl["map"] / rl["params_m"])
        print(f"{hi} vs {lo}: mAP比={m:.2f}  E_FLOPs余量={f_ratio:.2f}  "
              f"E_Lat余量={t_ratio:.2f}  E_Params余量={p_ratio:.2f}")

    # 选型协议示例（三档预算）
    print("\n== 选型协议示例（LAE 默认，预算约束下取最优）==")
    for tag, mn, ml, mp in [
        ("轻量实时 mAP≥0.37&≤40ms", 0.37, 40.0, float("inf")),
        ("中等精度 mAP≥0.50&≤200ms", 0.50, 200.0, float("inf")),
        ("高精度 mAP≥0.53", 0.53, float("inf"), float("inf")),
    ]:
        sel = select_best(rows, mn, ml, mp)
        print(f"{tag}: -> {sel}")

    # 落盘：逐点网格 + 默认/朴素排序
    Path("results").mkdir(exist_ok=True)
    with open("results/lae_sweep_grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "beta", "identical", "rho", "order"])
        for c in stab["cells"]:
            w.writerow([c["alpha"], c["beta"], int(c["identical"]),
                        f"{c['rho']:.4f}", c["order"]])
    with open("results/lae_sweep_baselines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "order", "rho_vs_default", "identical"])
        defs = default_order
        for name, order in baselines.items():
            rho = rank_vs_default(rank_positions(order), default_pos)
            w.writerow([name, ">".join(order), f"{rho:.4f}", int(order == defs)])
    print("\n[saved] results/lae_sweep_grid.csv, results/lae_sweep_baselines.csv")


if __name__ == "__main__":
    main()
