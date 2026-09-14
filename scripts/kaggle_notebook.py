"""生成轻量子集跨平台验证 notebook（Y 挡；Kaggle 与 ModelScope 双平台通用）。

目的：在第 2 台**纯 CPU** 平台（Kaggle 免费 CPU 或 ModelScope 魔搭免费 CPU，绝不用 GPU）
上测 YOLO11n / YOLOv8n/m/l 四个 ONNX 的**前向**延迟，验证本地（i7-14650HX）得到的
效率排序能否跨平台重现。

为什么"只测前向、免 NMS"（2026-09-07 用户拍板）：
- 子集全为 YOLO 族，本地"前向序 == 端到端序"（NMS 恒 ~2ms 平移），跨平台验的是**序**；
- notebook 代码最小、不移植 decode/NMS，云端少一个出错面。
时长档：标准（warmup 3 × 3 图 × reps 10 → 中位），8 核约 3-6 分钟。
文件：4 个 .onnx 放 ModelScope 的 /mnt/workspace（或 Kaggle 的 Dataset）；resolver 按候选目录
递归查找（/mnt/workspace → /kaggle/input → 当前目录），CSV 存到第一个可写目录。

关键纪律：
- 本 notebook 里 mAP/params 用**模型固有值**（CPU 无关，账本 §1/§2），不在云端重测精度；
- 跨平台**只比排序**，不比绝对值（云端未知频率/共享 CPU，绝对延迟不可跨台比）；
- 判定 = 测得的 fwd 排序 与 本地 canonical fwd 排序逐点比对（PASS/FAIL）。

用法：
    python scripts/kaggle_notebook.py            # 生成 kaggle/Y_cross_cpu_verify.ipynb
    python scripts/kaggle_notebook.py --test     # 本地跑纯函数自检（等价 pytest 入口，写日志）
（Rule 3：本模块纯函数有单测 test_kaggle_notebook.py，测完即删。）
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_NB = ROOT / "kaggle" / "Y_cross_cpu_verify.ipynb"

# 轻量子集：name -> (onnx 文件名, 官方 mAP, 参数 M, GFLOPs, 本地 canonical fwd 中位 ms)
# 值来自账本 §1/§2（模型固有 + 2026-09-07 规范同窗口表）。mAP 官方 COCO val2017。
SUBSET = [
    {"name": "YOLO11n", "file": "yolo11n.onnx", "map": 0.395, "params_m": 2.658,
     "gflops": 6.54, "local_fwd_ms": 27.447},
    {"name": "YOLOv8n", "file": "yolov8n.onnx", "map": 0.373, "params_m": 3.194,
     "gflops": 8.74, "local_fwd_ms": 29.321},
    {"name": "YOLOv8m", "file": "yolov8m.onnx", "map": 0.502, "params_m": 25.928,
     "gflops": 78.94, "local_fwd_ms": 149.291},
    {"name": "YOLOv8l", "file": "yolov8l.onnx", "map": 0.529, "params_m": 43.710,
     "gflops": 165.15, "local_fwd_ms": 270.555},
]
IMG = 640
ALPHA, BETA = 0.5, 0.3


# ---- 供 notebook 嵌入 + 本地单测的纯函数（保持自包含，不 import 项目模块）----

def make_inputs(n: int = 3, seed: int = 0, img: int = IMG) -> list:
    """造 n 个 1x3ximgximg 的确定性 float32 输入（时序基准不需要真实图语义）。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((1, 3, img, img), dtype=np.float32) for _ in range(n)]


def median_ms(times: list[float]) -> float:
    import numpy as np
    return float(np.median(times))


def measure_forward(session, input_name: str, outputs: list[str],
                    inputs: list, warmup: int, reps: int) -> list[float]:
    """前向延迟：warmup 不计数；每个输入各 reps 次，逐次计时(s.run) → ms 列表。"""
    import time
    for _ in range(warmup):
        session.run(outputs, {input_name: inputs[0]})
    times = []
    for x in inputs:
        for _ in range(reps):
            t0 = time.perf_counter()
            session.run(outputs, {input_name: x})
            times.append((time.perf_counter() - t0) * 1e3)
    return times


def build_session(path: str, intra_threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, intra_threads)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def resolve_onnx(files: list[str], roots: list[str]) -> dict[str, str]:
    """在候选目录列表 roots 里递归找每个 onnx（/mnt/workspace、/kaggle/input、当前目录
    都放进列表即可同时兼容 ModelScope 与 Kaggle）。找不到的文件不出现在返回 dict，
    由调用方报可读错误。"""
    found: dict[str, str] = {}
    for f in files:
        for r in roots:
            if not (r and os.path.isdir(r)):
                continue
            hit = None
            for dirpath, _, names in os.walk(r):
                if f in names:
                    hit = os.path.join(dirpath, f)
                    break
            if hit:
                found[f] = hit
                break
    return found


def lae_score(map_: float, lat_ms: float, params_m: float,
              alpha: float = ALPHA, beta: float = BETA) -> float:
    return map_ / (lat_ms ** alpha * params_m ** beta)


def ordered_names(data: list[dict], lat_key: str,
                  score_key: str | None = None) -> list[str]:
    """按 LAE（若给 score_key 先算分）降序 → 名字列表；否则按 lat_key 升序（LAE 对固定
    mAP/params 只随延迟单调，lat 升序即 LAE 降序）。统一用名次并列处理（这里不会并列）。"""
    if score_key is not None:
        order = sorted(data, key=lambda d: -d[score_key])
    else:
        order = sorted(data, key=lambda d: d[lat_key])
    return [d["name"] for d in order]


# ---- notebook 源码片段（把上面的函数嵌进单元格，保证本地测的==Kaggle 跑的）----

def _source() -> str:
    import inspect
    fns = [make_inputs, median_ms, measure_forward, build_session,
           resolve_onnx, lae_score, ordered_names]
    return "\n\n".join(inspect.getsource(f) for f in fns)


def _subset_literal() -> str:
    return json.dumps(SUBSET, ensure_ascii=True)


def build_notebook() -> Path:
    md_title = (
        "# Y 挡 · 跨平台验证：轻量子集 (YOLO11n / YOLOv8n/m/l) CPU 前向排序\n\n"
        "**目的**：本机 i7-14650HX（16 线程）测得 YOLO 轻量子集 fwd 排序为\n"
        "`YOLO11n < YOLOv8n < YOLOv8m < YOLOv8l`（账本 §2 规范同窗口表）。\n"
        "本 notebook 在第 2 台**纯 CPU** 平台重测同一批 ONNX 的**前向**延迟，验证排序能否跨平台重现。\n"
        "已适配两个平台，任选其一：**ModelScope 魔搭免费 CPU**（国内免代理）或 **Kaggle 免费 CPU**。\n\n"
        "**口径**：只测前向（ORT `session.run`），免 NMS/解码（子集全为 YOLO，本地前向序=端到端序）；\n"
        "warmup 3 × 3 张确定性输入 × reps 10 → 取中位；约 3-6 分钟。\n\n"
        "**纪律**：跨平台**只比排序不比绝对值**（云端未知频率共享 CPU）；mAP/params 用模型固有值\n"
        "（不在此重测精度）。判定=测得 fwd 排序与本地逐点比对 → PASS/FAIL。\n\n"
        "**跑完回报**：把输出的 CSV（路径见最后一格打印；通常 `/mnt/workspace/kaggle_fwd.csv` 或\n"
        "`/kaggle/working/kaggle_fwd.csv`）下载后发回项目，我会记进 `results/data_ledger.md`（Y 挡 §）\n"
        "并给出跨平台秩一致性结论。"
    )

    md_run = (
        "## 怎么用（两个平台二选一）\n"
        "- **ModelScope（魔搭，国内免代理，推荐）**：启动**免费 CPU 环境** → JupyterLab 打开本 notebook →\n"
        "  把 4 个 .onnx（`yolo11n / yolov8n / yolov8m / yolov8l.onnx`，共约 290MB）上传到左侧文件区\n"
        "  `/mnt/workspace/`（慢就打包成一个 zip 上传，在终端 `unzip`；放子目录也会被递归找到）→ Run All。\n"
        "- **Kaggle**：把 4 个 .onnx 打成 Dataset 并 Add Input 挂到本 notebook → 右上角确认\n"
        "  Accelerator 为 None/CPU → Run All。\n"
        "本 notebook 自动按候选目录找文件：`/mnt/workspace` → `/kaggle/input` → 当前目录。\n\n"
        "若报 `找不到 onnx`：检查四个文件名是否原样、是否在以上任一目录里。CSV 会存到第一个可写目录\n"
        "（通常是 `/mnt/workspace/kaggle_fwd.csv` 或 `/kaggle/working/kaggle_fwd.csv`）。"
    )

    cell_setup = (
        "import importlib, os, sys, subprocess, json, time\n"
        "import numpy as np\n\n"
        "# 常量提前定义：内嵌函数 make_inputs/lae_score 的默认参数依赖它们（勿删，否则 NameError）\n"
        "IMG = 640\n"
        "ALPHA, BETA = 0.5, 0.3\n\n"
        "# 确保 onnxruntime（Kaggle 通常自带；缺失才装）\n"
        "try:\n"
        "    import onnxruntime as ort\n"
        "except ModuleNotFoundError:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnxruntime'], check=True)\n"
        "    import onnxruntime as ort\n\n"
        "print('ORT', ort.__version__)\n"
        "print('CPU count:', os.cpu_count())\n"
        "print('Providers:', ort.get_available_providers())\n"
        "assert 'CPUExecutionProvider' in ort.get_available_providers(), '必须纯 CPU'\n"
        "n_threads = max(1, os.cpu_count() or 2)"
    )

    cell_helpers = _source()

    cell_data = (
        "# 轻量子集：模型固有值(官方 mAP / 参数量) + 本机 fwd 参照（账本 §1/§2，勿改）\n"
        "SUBSET = " + _subset_literal() + "\n"
        "# 文件候选目录（按序查找）：ModelScope 持久目录 /mnt/workspace → Kaggle /kaggle/input → 当前目录\n"
        "ROOTS = ['/mnt/workspace', '/kaggle/input', '.']"
    )

    cell_measure = (
        "files = [d['file'] for d in SUBSET]\n"
        "found = resolve_onnx(files, ROOTS)\n"
        "missing = [f for f in files if f not in found]\n"
        "if missing:\n"
        "    raise SystemExit(f'找不到 onnx: {missing}。请把 4 个 .onnx 上传到 ModelScope 的 /mnt/workspace，"
        "或（Kaggle）挂到 Dataset attach。')\n\n"
        "inputs = make_inputs(n=3, seed=0)\n"
        "for d in SUBSET:\n"
        "    sess = build_session(found[d['file']], n_threads)\n"
        "    inp = sess.get_inputs()[0].name\n"
        "    outs = [o.name for o in sess.get_outputs()]\n"
        "    # 冒烟：确认模型真在跑（输出形状非空）\n"
        "    probe = sess.run(outs, {inp: inputs[0]})[0]\n"
        "    assert probe.size > 0, f'{d[\"name\"]} 输出为空'\n"
        "    ts = measure_forward(sess, inp, outs, inputs, warmup=3, reps=10)\n"
        "    d['kaggle_fwd_ms'] = median_ms(ts)\n"
        "    d['fps'] = 1000.0 / d['kaggle_fwd_ms']\n"
        "    print(f\"{d['name']:9s} fwd中位={d['kaggle_fwd_ms']:8.1f} ms  fps={d['fps']:6.2f}\", flush=True)"
    )

    cell_verdict = (
        "# 排序比对：实测 vs 本地参照；LAE 分数（mAP/(L^α·P^β)，α=0.5 β=0.3，账本 §5 口径）\n"
        "for d in SUBSET:\n"
        "    d['lae'] = lae_score(d['map'], d['kaggle_fwd_ms'], d['params_m'])\n"
        "\n"
        "order_kaggle = ordered_names(SUBSET, 'kaggle_fwd_ms')\n"
        "order_local = ordered_names(SUBSET, 'local_fwd_ms')\n"
        "order_lae = ordered_names(SUBSET, 'kaggle_fwd_ms', 'lae')\n"
        "ok = order_kaggle == order_local\n"
        "print('本地参照 fwd 排序 :', ' < '.join(order_local))\n"
        "print('Kaggle 实测 fwd 排序:', ' < '.join(order_kaggle))\n"
        "print('Kaggle LAE 降序     :', ' > '.join(order_lae))\n"
        "print('\\n判定:', 'PASS —— 跨平台秩一致' if ok else 'FAIL —— 秩不一致，需查证')\n\n"
        "# 存表回传（模型固有 + 实测 fwd）：选第一个可写目录 —— /mnt/workspace → /kaggle/working → 当前目录\n"
        "import csv\n"
        "out_dir = next(p for p in ['/mnt/workspace', '/kaggle/working', '.'] if os.path.isdir(p))\n"
        "csv_path = os.path.join(out_dir, 'kaggle_fwd.csv')\n"
        "with open(csv_path, 'w', newline='', encoding='utf-8') as f:\n"
        "    w = csv.DictWriter(f, fieldnames=['name', 'map_official', 'params_m', 'kaggle_fwd_ms', 'fps', 'lae'])\n"
        "    w.writeheader()\n"
        "    for d in SUBSET:\n"
        "        w.writerow({'name': d['name'], 'map_official': d['map'], 'params_m': d['params_m'],\n"
        "                    'kaggle_fwd_ms': round(d['kaggle_fwd_ms'], 2), 'fps': round(d['fps'], 2),\n"
        "                    'lae': round(d['lae'], 5)})\n"
        "print('已保存', csv_path, '—— 下载这个文件后发回项目。')"
    )

    md_interpret = (
        "## 怎么解读结果\n"
        "- 只要 `判定: PASS`（四个模型的 fwd 排序与本机一致）即达到 Y 挡目标：LAE 跨平台排序稳定。\n"
        "- 云端的**绝对毫秒数会与本机不同/更大**（共享、虚拟化 CPU），属预期，勿与本机比较绝对值。\n"
        "- 若有 FAIL：先重跑一次排除偶发；仍 FAIL 再逐模型看——但本机 3 模型隔日重测早已证明秩可复现\n"
        "  （账本 §2 修正①），跨平台偶发波动不至于翻转三个数量级的间隔（m/l 与本机差 5-10×）。"
    )

    nb_cells = []
    for typ, src in [("markdown", md_title), ("markdown", md_run),
                     ("code", cell_setup), ("code", cell_helpers),
                     ("code", cell_data), ("code", cell_measure), ("code", cell_verdict),
                     ("markdown", md_interpret)]:
        nb_cells.append({"cell_type": typ, "metadata": {},
                         "source": [s for s in src.splitlines(keepends=True)]})

    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
          "cells": nb_cells}
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return OUT_NB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="本地自检纯函数并打印关键断言结果")
    args = ap.parse_args()
    if args.test:
        import numpy as np  # noqa: F401
        assert [i.shape for i in make_inputs(3)] == [(1, 3, 640, 640)] * 3
        assert median_ms([1.0, 2.0, 3.0]) == 2.0
        # 参照顺序：本机 canonical fwd 升序应=预期排序
        exp = ordered_names(SUBSET, "local_fwd_ms")
        assert exp == ["YOLO11n", "YOLOv8n", "YOLOv8m", "YOLOv8l"], exp
        assert lae_score(0.5, 40.0, 20.0) == 0.5 / (40 ** 0.5 * 20 ** 0.3)
        print("[selftest] OK —— 纯函数断言全部通过")
    p = build_notebook()
    print(f"[built] {p}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
