"""实时频率流式采样（修正① 配套）：逐行带时间戳落盘，被 kill 也不丢已采数据。

为什么不用 cpu_freq.py 直接跑满时长：cpu_freq.py 只在跑满 --seconds 后才把全部采样
写盘。而测量脚本时长不确定（取决于模型/负载/睿频），若把 --seconds 固定成一个估计值，
要么早退(测量没测完采样已停 → 缺尾)，要么空转(测量结束后继续采空闲低睿频 → 把空闲
样本算进窗口，压低均值/抬高 std)。本工具流式逐行 flush + 每行带 ISO 时间戳：
- 用 TaskStop / kill 中止时，已采数据全部在盘上；
- 事后用 --summarize --until <测量结束时刻> 截断，得到「恰好测量窗口内」的频率。

复用 cpu_freq._powershell_sample（该采样器本身已 5/5 测试）与 cpu_freq.summarize 口径，
保证 mean±std(GHz) 与论文 §2 的表述一致。

用法（并行两个任务）：
    # 任务1：流式采样（不设 --seconds 就一直采，设了则到时自动退出兜底）
    python scripts/freq_log_stream.py stream --interval 2.0 --out results/freq_raw.csv
    # 任务2：测量完成后，把统计截到测量结束时刻
    python scripts/freq_log_stream.py summarize --in results/freq_raw.csv \
        --until 2026-09-06T12:34:56
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cpu_freq import _powershell_sample, summarize  # noqa: E402


def _iso_now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def stream(interval: float, out: str, max_seconds: float | None) -> int:
    """每 interval 秒采一次，写 'iso,mhz' 一行并立即 flush，直到被中止或到 max_seconds。

    返回采样点数。Get-Counter 本身耗时，实际周期 ≈ interval + 采样耗时。
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    deadline = (time.monotonic() + max_seconds) if max_seconds else None
    with open(out, "w", newline="", encoding="utf-8") as f:
        while deadline is None or time.monotonic() < deadline:
            try:
                v = _powershell_sample()
            except RuntimeError as e:
                print(f"[freq] warn: {e}", file=sys.stderr)
                v = float("nan")  # 写 nan 保留时间对齐，汇总时剔除
            f.write(f"{_iso_now()},{v:.1f}\n")
            f.flush()  # 每行落盘：kill 也不丢
            n += 1
            time.sleep(interval)
    return n


def _read_rows(path: str) -> tuple[list[str], np.ndarray]:
    """读 stream 写的 CSV → (时间戳列表, mhz 数组)。行格式异常则跳过该行。"""
    ts: list[str] = []
    vals: list[float] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue  # 容忍半行（kill 瞬间）
            try:
                vals.append(float(parts[1]))
                ts.append(parts[0])
            except ValueError:
                continue
    if not vals:
        raise ValueError(f"{path} 无有效采样")
    return ts, np.array(vals, dtype=np.float64)


def summarize_until(path: str, until: str | None) -> dict:
    """统计 path 中全部（或 --until 时刻前）样本。剔除 nan。字段同 cpu_freq.summarize。"""
    ts, allv = _read_rows(path)
    if until is not None:
        t_end = _dt.datetime.fromisoformat(until)
        mask = np.array([_dt.datetime.fromisoformat(t) < t_end for t in ts])
        allv = allv[mask]
    good = allv[~np.isnan(allv)]
    if good.size == 0:
        raise ValueError(f"{path}（until={until}）内无有效频率样本")
    s = summarize(good)
    s["kept"] = int(good.size)
    s["skipped_nan"] = int(np.isnan(allv).sum())
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_stream = sub.add_parser("stream")
    p_stream.add_argument("--interval", type=float, default=2.0)
    p_stream.add_argument("--out", required=True)
    p_stream.add_argument("--max-seconds", type=float, default=None, help="自动退出兜底")

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--in", dest="in_path", required=True)
    p_sum.add_argument("--until", default=None,
                       help="ISO 时刻；只统计该时刻前的样本（排测量后的空闲期）")
    args = ap.parse_args()

    if args.cmd == "stream":
        n = stream(args.interval, args.out, args.max_seconds)
        print(f"[freq-stream] 共采 {n} 点 -> {args.out}")
    else:
        s = summarize_until(args.in_path, args.until)
        print(f"[freq-summary] kept={s['kept']} (nan剔除{s['skipped_nan']}) "
              f"mean={s['mean_ghz']:.3f}±{s['std_ghz']:.3f} GHz "
              f"({s['min_mhz']:.0f}~{s['max_mhz']:.0f} MHz)")


if __name__ == "__main__":
    main()
