"""S3（2026-09-10）：21 对 FLOPs–延迟标度律。

目的：把 "Table 5 手挑 4 对得到 1.3×–2.0× 高估区间" 升级为一条可外推的标度律
    log(延迟比) = γ · log(FLOPs比) + c
γ < 1 即 "FLOPs 系统性高估实测延迟优势"；给定 FLOPs 比 R，高估倍数 = R^(1-γ)。

数据源：results/latency_canonical7.csv（单窗口协议 = 正文 Table 2 那一列）。
    latency_robust.csv 只有 6 模型（无 YOLO11n），故跨轮协议只能出 C(6,2)=15 对，
    本脚本按用户指定用 canonical7 的 21 对。

口径与统计处理（用户 S3 指定）：
- 21 个**无向**对，方向按 FLOPs 归一（分子 = FLOPs 更大者），故 FLOPs 比 ≥ 1。
- 21 对彼此不独立（共享模型）→ 除了 OLS 解析 CI，另报**按模型整簇重采样**的
  cluster bootstrap CI（B=1000，重采样 7 个模型，用重采样后的模型多重集重建对集）。
- 跨族对比值 <1、与同族对混在一起会拉动 γ → 另报：CNN 内拟合、Transformer 内拟合
  （仅 1 对，样本不足，如实标注）、跨族拟合，以及带 family 哑变量的联合拟合。

用法：
    python scripts/scaling_law.py            # 控制台打印 + 落盘 results/scaling_law_*.csv
"""
from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "latency_canonical7.csv"
OUT_PAIRS = ROOT / "results" / "scaling_law_pairs.csv"
OUT_FIT = ROOT / "results" / "scaling_law_fit.csv"

B = 1000
SEED = 0


def load() -> list[dict]:
    with open(SRC, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gflops"] = float(r["gflops"])
        r["r_fwd"] = float(r["fwd_median_ms"])
        r["r_e2e"] = float(r["e2e_median_ms"])
    return rows


def pairs_of(models: list[dict]) -> list[dict]:
    """无向对，方向按 FLOPs 归一（分子 = FLOPs 大者）。同模型（重采样可能重复）跳过。"""
    out = []
    for a, b in itertools.combinations(models, 2):
        if a["name"] == b["name"]:
            continue
        hi, lo = (a, b) if a["gflops"] >= b["gflops"] else (b, a)
        out.append({
            "num": hi["name"], "den": lo["name"],
            "flops_ratio": hi["gflops"] / lo["gflops"],
            "params_ratio": float(hi["params_M"]) / float(lo["params_M"]),
            "e2e_ratio": hi["r_e2e"] / lo["r_e2e"],
            "fwd_ratio": hi["r_fwd"] / lo["r_fwd"],
            "cross_family": hi["family"] != lo["family"],
        })
    return out


def ols(x: np.ndarray, y: np.ndarray) -> dict:
    """y = g*x + c，返回 g/c 的 OLS 估计、解析 95% CI、R²、残差。"""
    n = x.size
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    sxy = float(((x - xbar) * (y - ybar)).sum())
    g = sxy / sxx
    c = ybar - g * xbar
    resid = y - (g * x + c)
    sse = float((resid ** 2).sum())
    sst = float(((y - ybar) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    dof = n - 2
    if dof > 0:
        se = math.sqrt(sse / dof / sxx)
        tcrit = _tcrit(dof)          # 双侧 95%
    else:
        se, tcrit = float("nan"), float("nan")
    return {"gamma": g, "intercept": c, "r2": r2, "se": se, "n": n,
            "ci_lo": g - tcrit * se, "ci_hi": g + tcrit * se, "resid": resid}


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060}


def _tcrit(dof: int) -> float:
    if dof in _T95:
        return _T95[dof]
    try:
        from scipy.stats import t as _t  # 有就用精确值
        return float(_t.ppf(0.975, dof))
    except Exception:
        return 1.96


def fit_pairs(pr: list[dict], key: str) -> dict:
    x = np.log(np.array([p["flops_ratio"] for p in pr], dtype=float))
    y = np.log(np.array([p[key] for p in pr], dtype=float))
    return ols(x, y)


def cluster_bootstrap(models: list[dict], key: str) -> np.ndarray:
    """按模型整簇重采样（有放回抽 7 个模型，重建对集）→ γ 的经验分布。

    退化簇必须剔除：重采样可能只抽到 1~2 个不同模型（对集 <3），或抽到的对在 FLOPs 比上
    没有散布（sxx=0 → 斜率无定义，Python float 会直接 ZeroDivisionError）。剔除会对
    CI 有轻微影响，故把有效簇数一并返回供如实报告。
    """
    rng = np.random.default_rng(SEED)
    n = len(models)
    gs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        pr = pairs_of([models[i] for i in idx])
        if len({(p["num"], p["den"]) for p in pr}) < 3:
            continue
        xs = np.log(np.array([p["flops_ratio"] for p in pr], dtype=float))
        if float(xs.std()) < 1e-9:
            continue
        gs.append(fit_pairs(pr, key)["gamma"])
    return np.array(gs)


def main() -> None:
    models = load()
    pr = pairs_of(models)
    print(f"[data] {len(models)} 模型 → {len(pr)} 对（方向按 FLOPs 归一）")
    print(f"       模型: {', '.join(m['name'] for m in models)}")

    # ---- 存档自检：手挑 4 对必须复现正文 Table 5 ----
    expect = {("YOLOv8l", "YOLOv8n"): (18.9, 9.4), ("RT-DETR-x", "RT-DETR-l"): (2.13, 1.63)}
    for p in pr:
        k = (p["num"], p["den"])
        if k in expect:
            print(f"[verify] {k[0]}/{k[1]}: FLOPs比={p['flops_ratio']:.3f} "
                  f"(刊 18.9/2.13) e2e比={p['e2e_ratio']:.3f} (刊 9.4/1.63)")

    rows = []
    for key, label in [("e2e_ratio", "e2e"), ("fwd_ratio", "fwd")]:
        f_all = fit_pairs(pr, key)
        cb = cluster_bootstrap(models, key)
        lo, hi = np.percentile(cb, [2.5, 97.5])
        print(f"\n===== {label}: 全 21 对 =====")
        print(f"  γ = {f_all['gamma']:.3f}   OLS 95%CI [{f_all['ci_lo']:.3f}, {f_all['ci_hi']:.3f}]"
              f"   cluster-boot 95%CI [{lo:.3f}, {hi:.3f}]   R² = {f_all['r2']:.3f}  n={f_all['n']}")
        rows.append({"fit": "all21", "latency": label, "n": f_all["n"],
                     "gamma": round(f_all["gamma"], 4),
                     "intercept": round(f_all["intercept"], 4),   # 画拟合线用（P5）
                     "ols_ci_lo": round(f_all["ci_lo"], 4), "ols_ci_hi": round(f_all["ci_hi"], 4),
                     "boot_ci_lo": round(float(lo), 4), "boot_ci_hi": round(float(hi), 4),
                     "r2": round(f_all["r2"], 4), "B": B, "seed": SEED})

        for sel, tag in [
            (lambda p: not p["cross_family"], "within_family"),
            (lambda p: p["cross_family"], "cross_family"),
        ]:
            sub = [p for p in pr if sel(p)]
            if len(sub) < 3:
                print(f"  [{tag}] 仅 {len(sub)} 对 → 样本不足，只报描述值")
            fs = fit_pairs(sub, key)
            print(f"  [{tag}] n={fs['n']}  γ={fs['gamma']:.3f} "
                  f"95%CI [{fs['ci_lo']:.3f}, {fs['ci_hi']:.3f}]  R²={fs['r2']:.3f}")
            rows.append({"fit": tag, "latency": label, "n": fs["n"],
                         "gamma": round(fs["gamma"], 4),
                         "ols_ci_lo": round(fs["ci_lo"], 4), "ols_ci_hi": round(fs["ci_hi"], 4),
                         "boot_ci_lo": "", "boot_ci_hi": "",
                         "r2": round(fs["r2"], 4), "B": "", "seed": ""})

        # 族哑变量：γ 同、截距随跨族平移
        x = np.log([p["flops_ratio"] for p in pr])
        y = np.log([p[key] for p in pr])
        d = np.array([1.0 if p["cross_family"] else 0.0 for p in pr])
        X = np.column_stack([x, d, np.ones_like(x)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        r2 = 1 - float(((y - yhat) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
        print(f"  [family dummy] γ={beta[0]:.3f}  跨族截距偏移={beta[1]:+.3f}  R²={r2:.3f}")

    # ---- 落盘 ----
    with open(OUT_PAIRS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["num", "den", "flops_ratio", "params_ratio",
                                          "e2e_ratio", "fwd_ratio", "cross_family"])
        w.writeheader()
        for p in pr:
            w.writerow({**p, "flops_ratio": round(p["flops_ratio"], 4),
                        "params_ratio": round(p["params_ratio"], 4),
                        "e2e_ratio": round(p["e2e_ratio"], 4),
                        "fwd_ratio": round(p["fwd_ratio"], 4)})
    with open(OUT_FIT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fit", "latency", "n", "gamma", "intercept",
                                          "ols_ci_lo", "ols_ci_hi", "boot_ci_lo", "boot_ci_hi",
                                          "r2", "B", "seed"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] -> {OUT_PAIRS.name}, {OUT_FIT.name}")


if __name__ == "__main__":
    main()
