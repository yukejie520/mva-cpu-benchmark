"""Windows CPU 实测频率采样（修正①：监控睿频/温度墙，验证未触墙）。

为什么存在：账本 §0 论文声明要写「频率稳定在 X±Y GHz 未触墙」。Windows 下
psutil.cpu_freq() 返回的是**请求频率**而非实测（睿频激活时差 0.5-1.0 GHz，会骗人）。
正确来源是 Windows Performance Counter：路径 "Processor Information(_Total)\\
Processor Frequency"（本文件顶部 COUNTER_PATH），PowerShell `Get-Counter` 可读
（2026-09-06 实测返回 ~2114-2142 MHz）。本模块在测量期间并行采样，输出 mean±std，
供论文 §2 如实填写。

用法（配合测量脚本）：
    # 在测量脚本跑的同时，另开一个 shell：
    python scripts/cpu_freq.py --seconds 120 --interval 1.0 --out results/freq_log.csv
    # 单独快速测几秒：
    python scripts/cpu_freq.py --seconds 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

COUNTER_PATH = r"\Processor Information(_Total)\Processor Frequency"


def _powershell_sample(timeout: float = 10.0) -> float:
    """用 Get-Counter 读一次当前实测频率(MHz)。返回 float；失败抛 RuntimeError。"""
    cmd = [
        "powershell", "-NoProfile", "-Command",
        f"(Get-Counter '{COUNTER_PATH}' -MaxSamples 1).CounterSamples[0].CookedValue",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"Get-Counter 失败: {out.stderr.strip()}")
    try:
        return float(out.stdout.strip())
    except ValueError as e:
        raise RuntimeError(f"无法解析频率读数 '{out.stdout.strip()}'") from e


def sample_freq_mhz(seconds: float, interval: float = 1.0) -> np.ndarray:
    """每 interval 秒采一次频率，共采 ~seconds/interval 点。返回 (MHz,) 数组。"""
    if seconds <= 0:
        raise ValueError("seconds 必须 > 0")
    samples: list[float] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            samples.append(_powershell_sample())
        except RuntimeError as e:
            print(f"[freq] warn: {e}", file=sys.stderr)
        time.sleep(interval)
    return np.array(samples, dtype=np.float64)


def summarize(freqs_mhz: np.ndarray) -> dict:
    """把频率采样(MHz)汇总成报告用的 mean±std(MHz/GHz)、min/max。"""
    if freqs_mhz.size == 0:
        raise ValueError("无有效频率采样")
    return {
        "n": int(freqs_mhz.size),
        "mean_mhz": float(freqs_mhz.mean()),
        "std_mhz": float(freqs_mhz.std()),
        "min_mhz": float(freqs_mhz.min()),
        "max_mhz": float(freqs_mhz.max()),
        "mean_ghz": float(freqs_mhz.mean()) / 1000.0,
        "std_ghz": float(freqs_mhz.std()) / 1000.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, required=True, help="采样时长(秒)")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="CSV 输出路径（无则只打印汇总）")
    args = ap.parse_args()

    freqs = sample_freq_mhz(args.seconds, args.interval)
    s = summarize(freqs)
    print(f"[freq] {s['n']} samples: mean={s['mean_ghz']:.3f}±{s['std_ghz']:.3f} GHz "
          f"({s['min_mhz']:.0f}~{s['max_mhz']:.0f} MHz)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            for v in freqs:
                f.write(f"{v:.1f}\n")
        print(f"[freq] raw samples -> {args.out}")


if __name__ == "__main__":
    main()
