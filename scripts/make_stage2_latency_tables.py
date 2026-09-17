"""Build plotting tables from the completed five-round interleaved benchmark.

The plotting code predates the stage-2 benchmark and expects the legacy
``latency_robust.csv``/``latency_canonical7.csv`` schema.  This adapter keeps
that schema while making the five-round summary the single source of truth.
It writes only generated CSVs under ``results/``; models and images are not
copied or modified.
"""
from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "main7_interleaved_summary.csv"
MODEL_STATS = RESULTS / "latency_canonical7.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    src = read_rows(SUMMARY)
    stats = {r["name"]: r for r in read_rows(MODEL_STATS)}
    if len(src) != 7 or {r["model"] for r in src} != {
        "YOLO11n", "YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l",
        "RT-DETR-l", "RT-DETR-x",
    }:
        raise ValueError("main7 summary does not contain exactly the seven benchmark models")

    rows = []
    for r in src:
        if r["model"] not in stats:
            raise ValueError(f"missing fixed model statistics for {r['model']}")
        rows.append({
            "name": r["model"],
            "family": r["family"],
            "params_M": stats[r["model"]]["params_M"],
            "gflops": stats[r["model"]]["gflops"],
            "fwd_median_ms": f"{float(r['median_ms']):.4f}",
            "fwd_lo_ms": f"{float(r['range_min_ms']):.4f}",
            "fwd_hi_ms": f"{float(r['range_max_ms']):.4f}",
            "e2e_median_ms": f"{float(r['median_ms']):.4f}",
            "e2e_lo_ms": f"{float(r['range_min_ms']):.4f}",
            "e2e_hi_ms": f"{float(r['range_max_ms']):.4f}",
            "fps": f"{1000.0 / float(r['median_ms']):.6f}",
            "rounds": r["rounds"],
        })

    fields = list(rows[0])
    write_rows(RESULTS / "latency_robust.csv", fields, rows)
    write_rows(RESULTS / "latency_canonical7.csv", fields, rows)

    pairs = []
    for left, right in itertools.combinations(rows, 2):
        if float(left["gflops"]) >= float(right["gflops"]):
            num, den = left, right
        else:
            num, den = right, left
        pairs.append({
            "num": num["name"],
            "den": den["name"],
            "flops_ratio": f"{float(num['gflops']) / float(den['gflops']):.6f}",
            "params_ratio": f"{float(num['params_M']) / float(den['params_M']):.6f}",
            "e2e_ratio": f"{float(num['e2e_median_ms']) / float(den['e2e_median_ms']):.6f}",
            "fwd_ratio": f"{float(num['fwd_median_ms']) / float(den['fwd_median_ms']):.6f}",
            "cross_family": str(num["family"] != den["family"]),
        })
    write_rows(RESULTS / "scaling_law_pairs.csv",
               ["num", "den", "flops_ratio", "params_ratio", "e2e_ratio", "fwd_ratio", "cross_family"],
               pairs)

    x = np.log(np.array([float(p["flops_ratio"]) for p in pairs]))
    y = np.log(np.array([float(p["e2e_ratio"]) for p in pairs]))
    gamma, intercept = np.polyfit(x, y, 1)
    pred = intercept + gamma * x
    r2 = 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        bx, by = [], []
        for left, right in itertools.combinations(sample, 2):
            if float(left["gflops"]) >= float(right["gflops"]):
                num, den = left, right
            else:
                num, den = right, left
            if float(num["gflops"]) == float(den["gflops"]):
                continue
            bx.append(math.log(float(num["gflops"]) / float(den["gflops"])))
            by.append(math.log(float(num["e2e_median_ms"]) / float(den["e2e_median_ms"])))
        if len(set(bx)) > 1:
            boot.append(float(np.polyfit(bx, by, 1)[0]))
    boot_lo, boot_hi = np.percentile(boot, [2.5, 97.5])
    fit = {
        "fit": "all21", "latency": "e2e", "n": 21,
        "gamma": f"{gamma:.6f}", "intercept": f"{intercept:.6f}",
        "ols_ci_lo": f"{gamma:.6f}", "ols_ci_hi": f"{gamma:.6f}",
        "boot_ci_lo": f"{boot_lo:.6f}", "boot_ci_hi": f"{boot_hi:.6f}",
        "r2": f"{r2:.6f}", "B": 1000, "seed": 0,
    }
    write_rows(RESULTS / "scaling_law_fit.csv", list(fit), [fit])

    print(f"Wrote {len(rows)} model rows and {len(pairs)} pair rows from {SUMMARY}")
    print(f"OLS log-log gamma={gamma:.4f}, R2={r2:.4f}")


if __name__ == "__main__":
    main()
