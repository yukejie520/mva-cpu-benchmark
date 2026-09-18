"""论文图 P1-P4 生成（W5–W6 迭代 v5；数据全部从 results/*.csv 单源读入，
不手抄账本）。

v5 改动（2026-09-07 新增 P4，用户拍板 双面板）：
- 新增 plot_p4：P4 = (a) LAE α×β 秩稳健热图（宽区 240 点：绿色=7 模型排序与默认 (0.5,0.3) 恒同、
  粉/白=翻转；仅出现在 α≤0.15∧β≤0.1 退化角）+ (b) 效率余量失真柱对（FLOPs 效率 vs 实测延迟效率）。
- P4(a) 网格在代码内用 lae_sweep.grid_stability 现算（与账本 §5.1 同源同函数）；(b) 的余量数字
  现场从 canonical CSV 重算（v8n/v8l 13.3 vs 6.1、v8l/RT-l 0.64 vs 0.83）→ 均不与账本手抄。
- 图注沿用 v4.2 纪律：手动短行 + 底部 padding，防水平截断/贴边；无 U+00F7 字形。

v5.1 改动（2026-09-07 用户对 P4 (b) 的反馈）：
- (b) parity=1 虚线：由细灰（#555555, lw0.9, 底层 zorder1）强化为深色 #111111、lw1.3、zorder3.6
  置顶并纳入图例（label="Parity (margin = 1)"）；图例移右上空白角，避免盖第一对高柱。
- 图例标签改语义：Theoretical — mAP / GFLOPs、Measured CPU — mAP / CPU ms。
- 图注 (b) 开头点破色码：magenta = 理论效率（mAP / GFLOPs），blue = 实测 CPU 效率
  （mAP / measured CPU ms）；并点明虚线含义（above = 第一个模型更高效，below = 第二个）。

v4 改动（2026-09-07 第三轮用户审图反馈，仅 P1）：
- P1 配色：柱色 = 分子(A)所在族（CNN 蓝=bar1-2、Transformer 橙=bar3-4，与 P2 同调）；第 4 根
  （跨族 RT-DETR-l / YOLOv8l，三轴全 <1）加粗边 + 红粗数值 + x 轴星号三重高亮（克制不加新色）；
  删每面板 "ratio = 1" 旁注 → 图例集中放 (a) 右上（族色两块 + "-- Parity (ratio = 1)"）；
  柱顶数值加粗并上移不贴柱；(c) 标题含 16-thread i7-14650HX；x 轴改单行全名留白；图注压成 3 句。

v3 改动（2026-09-07 第二轮外部审图意见 + 用户拍板）：
- 全局：图内不再用 U+00F7 除号字形（部分阅读器渲染成 "+"），一律写成 "A / B" 比值措辞。
- P1：ylabel 改 "Ratio (A / B)"（A=对中第一个模型）；ratio=1 虚线上直接标注；(c) 延迟面板加
  跨轮区间折成的比值 whisker（a/b 为模型固有单值，无 whisker，如实）。见 plot_p1。
- P2：X 轴改 log（左端 YOLO11n/v8n 两点不再压成一团）；点旁标参数量数字，面积仍 ∝ √params
  （图例注明）；图注澄清星标 YOLO11n 在前沿最左端并支配 YOLOv8n；底部补硬件/NMS/单窗口口径脚注。
- P3：(b) 加 y 下限余量（triangles 不贴轴）；失败原因补 "in ORT v1.28.0 toolkit"（规避"是你们的
  calibration 写错"的质疑，如实是工具链数值失败）；(b) 脚注补 FP32 绝对 mAP 区间作基线参照。

v2 改动（2026-09-07 按用户目检反馈重画，消除了所有"颜色/刻度的歧义"，图要自说明）：
- P1 拆成三个小面板（FLOPs / Params / Latency 各一），每个面板只有一种指标 → 比值
  含义由面板标题直接给出，不再依赖"看图例认颜色"；log 轴在标题注明；数值标在柱顶。
- P2 图例写全：CNN/Transformer 族色、灰色空心=被支配、虚线=Pareto 最优集、红星=LAE top-1、
  点大小=参数量（代理尺寸+标题说明）；模型名逐个手摆偏移避免重叠/出界；LAE top-1 加文字高亮。
- P3 (a) 柱顶标真实体积比；(b) 每个点标真实 ΔmAP 值（消除把 Y 轴刻度当数据点的误读）；
  RT-DETR static 崩溃统一灰色 ✕ + 文字给原因（naive-static QDQ 数值崩溃，scale=0/NaN）。

统计口径（figure-statistics 纪律，详见 notes/07 图注）：
- 延迟点/柱 = 3 轮中位数之中位数（每轮 8 真实图 × 30 reps = 240 样本 → 中位）。
- mAPΔ = 同一 500 图 COCO 子集单次评估 INT8−FP32，无重测 → 单点无误差条，不画显著性星号。
- 体积比 = 量化产物 MB / FP32 MB。

用法：python scripts/plot_papers.py [--out results/figures]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import matplotlib.ticker as mpl_ticker  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from lae_sweep import (DEFAULT_ALPHA, DEFAULT_BETA, grid_stability, lae_score,  # noqa: E402
                       load_canonical, pareto_dominated)

# ---- 色彩纪律 ----
CNN_BLUE, TRANS_ORANGE = "#0072B2", "#E69F00"   # 族固定色（贯穿）
# 被支配 / 不可用。2026-09-11 由 #AAAAAA 调深到 #6E6E6E：
#   #AAAAAA 的 WCAG rel-lum 0.4020 → 灰度 170.0/255，与 TRANS_ORANGE(#E69F00, 172.7) 只差 2.7/255，
#   去色后二者**同灰**。P3 里 ✕ 标记正落在橙色柱旁，颜色通道在该处完全失效，只剩标记形状在兜底。
#   #6E6E6E 灰度 110.0/255，与橙的间距 62.7/255，与白底对比度 5.10:1。
#   注意：单一灰**不可能**同时远离蓝(108.9)和橙(172.7)——两者相距仅 63.8，任何灰的余量上限是 31.9。
#   这里选择最大化与**相邻**填充色（橙）的间距；蓝在 P3 (a) 中隔着 5 个柱组，不相邻。
DOM_GRAY = "#6E6E6E"
STAR_RED = "#D00000"                            # LAE top-1 高亮
METRIC_COLORS = {"FLOPs": "#CC79A7", "Params": "#999999", "Latency": "#56B4E9"}
RECIPE_MARKER = {"dynamic": "o", "naive": "^", "selective": "s"}  # 配方→点形
RECIPE_HATCH = {"dynamic": "", "naive": "//", "selective": "xx"}  # 配方→柱纹理
RECIPE_LABEL = {"dynamic": "dynamic INT8", "naive": "naive static", "selective": "selective static"}
# ---- 上标纪律（2026-09-11 用户拍板：普查后一次性改）----
# 图内公式一律用 mathtext 真上标，不用 ASCII `^`，也不用 Unicode 上标字符
# （α / β / γ 没有可用的 Unicode 上标，硬凑会造成混排，比 `^` 更糟）。
# mathtext.default="regular"：让 $...$ 内的字符取**正体常规字体**而非数学斜体，
#   从而与图内其余文字同族——否则一个标题里会出现两种字体，那是比 `^` 更显眼的新不一致。
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"   # 与图内正文同族（DejaVu Sans）
matplotlib.rcParams["mathtext.default"] = "regular"      # 正体，不倾斜

# ---- PDF 字体嵌入方式（2026-09-12 用户拍板）----
# matplotlib 的 pdf 后端默认 fonttype=3：字形以 **Type 3 的 glyph procedure** 内嵌。
# Springer 的 artwork 条款原文只要求 "Vector graphics containing fonts must have the fonts
# embedded in the files"，Type 3 算不算"embedded"是模糊的（本项目未能核到官方明文，
# 所以当时没有把它当缺陷报，而是列为"可选去模糊"）。改为 42 = **TrueType 子集嵌入**，
# 那是条款字面意义上的 embedded，歧义消失。代价：5 张图件的 PDF md5 全变（预期内），
# 因此这一次**不能**再用"未受影响的图逐字节不变"作验收——改为核字体子类型与页面尺寸。
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# 上标片段提到模块级：这些字符串要嵌进 f-string，而 mathtext 的 \alpha 带花括号，
# 在 f-string 里必须写成 {{ }}，写错一次就是静默的排版错。提到这里，调用处只剩 {常量}。
SUP_ALPHA = r"$^{\alpha}$"
SUP_BETA = r"$^{\beta}$"
SUP_GAMMA = r"$^{\gamma}$"
SUP_1MGAMMA = r"$^{(1-\gamma)}$"

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
OFFICIAL_MAP_CSV = ROOT / "results" / "full_coco_map_official.csv"
# MVA requires captions outside artwork. Existing footer calls use Figure.text;
# suppress only those calls while retaining axes labels and legends.
Figure.text = lambda self, *args, **kwargs: None


# ---------- 纯数据装载（可测试） ----------

def _read(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_plot_canonical(path: str | Path) -> list[dict]:
    """Join canonical latency with the validated local standard COCOeval table."""
    rows = load_canonical(path)
    with OFFICIAL_MAP_CSV.open(newline="", encoding="utf-8") as f:
        maps = {r["name"]: float(r["map50_95"]) for r in csv.DictReader(f)}
    if set(maps) != set(r["name"] for r in rows):
        raise ValueError(f"official map table does not cover the canonical latency models: {OFFICIAL_MAP_CSV}")
    for row in rows:
        row["map"] = maps[row["name"]]
    return rows


def load_robust(path: str | Path) -> dict[str, dict]:
    """读多轮稳健表 → {name: row}（P1 用：跨轮中位之中位数 = 官方正式值，账本 §2/§4）。"""
    return {r["name"]: r for r in _read(path)}


def ratios_p1(robust: dict[str, dict]) -> list[dict]:
    """4 个对比对的 FLOPs/Params/e2e 延迟比（num/den，num 在前）。
    比值 = 分子模型值 / 分母模型值；账本 §4 只要求这 4 对。"""
    pairs = [("YOLOv8l", "YOLOv8n"), ("YOLOv8s", "YOLOv8n"),
             ("RT-DETR-x", "RT-DETR-l"), ("RT-DETR-l", "YOLOv8l")]
    out = []
    for num, den in pairs:
        a, b = robust[num], robust[den]
        out.append({
            "pair": f"{num}/{den}", "family": "in" if num.startswith("YOLO") == den.startswith("YOLO") else "cross",
            "flops": float(a["gflops"]) / float(b["gflops"]),
            "params": float(a["params_M"]) / float(b["params_M"]),
            "latency": float(a["e2e_median_ms"]) / float(b["e2e_median_ms"]),
        })
    return out


def volume_fractions(path_naive: str | Path, path_sel: str | Path,
                     path_dyn: str | Path) -> dict[str, dict[str, float]]:
    """每模型每配方体积 / 该模型 FP32(in_MB)。scheme 归并为 dynamic/naive/selective。
    排除 RT-DETR 的 naive static：量化文件存在(体积可测)但推理数值崩溃不可用 → 图/表不算。"""
    scheme_of = {"dynamic": "dynamic", "static": "naive", "static_sel": "selective"}
    frac: dict[str, dict[str, float]] = {}
    for path, recipe in [(path_dyn, "dynamic"), (path_naive, "naive"), (path_sel, "selective")]:
        for r in _read(path):
            if scheme_of.get(r["scheme"]) != recipe:
                continue  # 行 scheme 必须与本配方一致，避免同文件里两 scheme 混写
            if recipe == "naive" and r["name"].startswith("RT-DETR"):
                continue  # 崩溃产物，不算可用配方
            frac.setdefault(r["name"], {})[recipe] = float(r["out_MB"]) / float(r["in_MB"])
    return frac


_CANON_NAMES = ["YOLO11n", "YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l",
                "RT-DETR-l", "RT-DETR-x"]


def _split_recipe(row_name: str) -> tuple[str, str]:
    """把 int8 行名拆成 (模型名, 配方)。命名不规整：YOLOv8n-yolov8n_static、
    RT-DETR-l-dynamic、YOLOv8n-static_sel。先剥尾部配方后缀，再按 7 个规范名前缀匹配。
    （RT-DETR-l/x 自带连字符，不能按第一个 '-' 切；YOLO 行还有 '-别名' 段。）"""
    if row_name.endswith("static_sel"):
        base, recipe = row_name[:-len("static_sel")], "selective"
    elif row_name.endswith("dynamic"):
        base, recipe = row_name[:-len("dynamic")], "dynamic"
    elif row_name.endswith("_static"):
        base, recipe = row_name[:-len("_static")], "naive"
    else:
        raise ValueError(f"无法解析配方: {row_name}")
    for m in _CANON_NAMES:
        if base == m or base.startswith(m + "-"):
            return m, recipe
    raise ValueError(f"行名不能归到规范模型: {row_name} (base={base})")


def mAP_deltas(base_csv: str | Path, int8_csv: str | Path) -> list[dict]:
    """500 子集 FP32→INT8 的 mAPΔ（同一子集单次评估）。静态 N/A 行不产生（需在图中显式加 ✕）。"""
    base = {r["name"]: float(r["map50_95"]) for r in _read(base_csv)}
    out = []
    for r in _read(int8_csv):
        model, recipe = _split_recipe(r["name"])
        if model not in base:
            continue
        out.append({"model": model, "recipe": recipe,
                    "delta": float(r["map50_95"]) - base[model]})
    return out


# ---- 面板 (c) 的 2×2：出口格式 × 配方 ----
# 顺序由用户拍板（QDQ-naive / QDQ-保头 / QOp-naive / QOp-保头），不按数值排。
# 元组 = (交错摘要里的变体键, 配对 CI 表里的 scheme 键, 图例用的格式名, 配方名)
INT8_CELLS = [
    ("QDQ_naive", "naive",   "QDQ",       "naive"),
    ("QDQ_sel",   "sel",     "QDQ",       "selective"),
    ("QOp_naive", "qop",     "QOperator", "naive"),
    ("QOp_sel",   "qop_sel", "QOperator", "selective"),
]
INT8_CELL_LABEL = {"QDQ_naive": "QDQ\nnaive", "QDQ_sel": "QDQ\nhead-pres.",
                   "QOp_naive": "QOperator\nnaive", "QOp_sel": "QOperator\nhead-pres."}
# 只有这两个尺度跑了 QOperator（m/l 没跑、RT-DETR 没有可用的 QOp），
# 所以面板 (c) 只能画这两个，图注必须写明，否则读者会外推。
INT8_MODELS = ["YOLOv8n", "YOLOv8s"]


def int8_round_matched(path: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """读交错重测摘要 → {model: {variant: {ratio, lo, hi}}}。

    这里的比值是**轮内配对**的（同一轮里 FP32 中位延迟 / 该变体中位延迟，逐轮算再取中位），
    基线与被测项共享同一热态，所以可以直接并排画。

    硬约束：**不能**改用 int8_latency_window.csv 里那条不同窗口的 FP32 基线。两次跑把
    YOLOv8n 的 FP32 中位放在 29.8 / 32.3 ms，差 8%，跨窗口拼出来的比值不是任何一次
    测量的结果（正文 §4.4 第一段就是在解释这件事）。
    """
    # 摘要 CSV 每行是「模型内某个变体」，模型名不在行里，由文件名承载（小写）。
    # 规范化成全篇统一的写法，否则返回的键是 "yolov8n"，和 INT8_MODELS 对不上。
    stem = Path(path).stem                      # interleaved_int8_yolov8n_summary
    token = stem.replace("interleaved_int8_", "").replace("_summary", "")
    canon = {m.lower(): m for m in INT8_MODELS}
    if token not in canon:
        raise ValueError(f"{Path(path).name} 的文件名对应不到已知模型：{token!r}；"
                         f"面板 (c) 只画 {INT8_MODELS}")
    model = canon[token]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for r in _read(path):
        if r["variant"] == "FP32":
            continue                            # 基线自身 ratio 恒为 1，不占柱子
        out.setdefault(model, {})[r["variant"]] = {
            "ratio": float(r["ratio_median"]),
            "lo": float(r["ratio_min"]),
            "hi": float(r["ratio_max"]),
            "ms": float(r["median_ms"]),
        }
    return out


def int8_delta_map(path: str | Path) -> dict[tuple[str, str], float]:
    """读配对 bootstrap 表 → {(model, scheme): delta_obs}，即该格子的 ΔmAP@50-95。"""
    return {(r["model"], r["scheme"]): float(r["delta_obs"]) for r in _read(path)}


def int8_panel_c(ratios: dict[str, dict[str, dict[str, float]]],
                 deltas: dict[tuple[str, str], float]) -> list[dict]:
    """把两份表按 2×2 拼成 8 根柱（2 模型 × 4 格），按模型、再按 INT8_CELLS 顺序展开。

    缺键直接报错，不静默跳过：少一根柱的 2×2 仍然"看起来像"一张图，
    靠肉眼是发现不了 QOperator 那一列整列没画上去的。
    """
    rows = []
    for m in INT8_MODELS:
        if m not in ratios:
            raise KeyError(f"交错摘要里没有 {m}；面板 (c) 只画 {INT8_MODELS}")
        for vkey, skey, fmt, recipe in INT8_CELLS:
            if vkey not in ratios[m]:
                raise KeyError(f"{m} 的交错摘要缺少变体 {vkey}")
            if (m, skey) not in deltas:
                raise KeyError(f"配对 CI 表缺少 ({m}, {skey})")
            rows.append({"model": m, "variant": vkey, "format": fmt, "recipe": recipe,
                         "delta": deltas[(m, skey)], **ratios[m][vkey]})
    return rows


def pareto_front(rows: list[dict]) -> list[str]:
    """按 (map↑, lat↓, params↓) 的非支配集，按延迟升序 → 用于画前沿连线。"""
    dom = pareto_dominated(rows)
    names = [r["name"] for r in rows if r["name"] not in dom]
    return sorted(names, key=lambda n: next(r for r in rows if r["name"] == n)["lat_ms"])


# ---------- 绘图 ----------

def _family(name: str) -> str:
    return "CNN" if name.startswith("YOLO") else "Transformer"


def _fcolor(name: str) -> str:
    return CNN_BLUE if _family(name) == "CNN" else TRANS_ORANGE


def _save(fig, out: Path) -> Path:
    for suf in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suf}"), dpi=300)
    plt.close(fig)
    return out.with_suffix(".png")


def plot_p1(robust_path: str | Path, out: Path) -> Path:
    """P1 v4：三面板（FLOPs / Params / e2e Latency），每面板 = 1 指标 × 4 对比对，log 柱。
    v4 改动（2026-09-07 用户审图反馈）：柱色 = 分子(A)所在族（蓝=YOLO、橙=RT-DETR，与 P2 同调）；
    第 4 根（跨族 RT-DETR-l / YOLOv8l，三轴全 <1）加粗边 + 红粗数值 + x 轴星号三重高亮；
    删每面板 "ratio = 1" 旁注 → 图例集中放 (a)（虚线 Parity + 族色两块）；柱顶数值加粗并上移；
    (c) 标题含 16-thread i7-14650HX；x 轴改单行全名留白；底部图注压成 3 句。延迟面板保留跨轮 whisker。"""
    robust = load_robust(robust_path)
    data = ratios_p1(robust)
    pairs = [("YOLOv8l", "YOLOv8n"), ("YOLOv8s", "YOLOv8n"),
             ("RT-DETR-x", "RT-DETR-l"), ("RT-DETR-l", "YOLOv8l")]
    x_names = ["YOLOv8l / YOLOv8n", "YOLOv8s / YOLOv8n",
               "RT-DETR-x / RT-DETR-l", "RT-DETR-l / YOLOv8l"]
    axes = [("flops", "FLOPs"), ("params", "Params"),
            ("latency", "Latency (e2e)")]  # (csv字段, 显示名)
    CROSS = 3  # 第 4 根 = 跨族全胜对（RT-DETR-l / YOLOv8l，三轴 ratio<1）
    n = len(data)
    x = np.arange(n)
    fig, axs = plt.subplots(3, 1, figsize=(6.9, 6.9), sharex=True,
                            gridspec_kw={"hspace": 0.34})
    letters = "abc"
    for k, (ax, (key, lab)) in enumerate(zip(axs, axes)):
        vals = [d[key] for d in data]
        colors = [CNN_BLUE if pairs[i][0].startswith("YOLO") else TRANS_ORANGE
                  for i in range(n)]          # 颜色 = 分子(A)所在族
        ews = [1.3 if i == CROSS else 0.4 for i in range(n)]  # 跨族柱加粗边
        bars = ax.bar(x, vals, 0.6, color=colors, edgecolor="#222222",
                      linewidth=ews, zorder=2)
        hi_txt = list(vals)
        if key == "latency":
            # 跨轮区间折成的比值 whisker：lo = num_lo/den_hi, hi = num_hi/den_lo
            for i, (num, den) in enumerate(pairs):
                a, b = robust[num], robust[den]
                lo = float(a["e2e_lo_ms"]) / float(b["e2e_hi_ms"])
                hi = float(a["e2e_hi_ms"]) / float(b["e2e_lo_ms"])
                ax.plot([x[i], x[i]], [lo, hi], color="black", lw=0.8, zorder=3)
                ax.plot([x[i] - 0.08, x[i] + 0.08], [lo, lo], color="black", lw=0.8)
                ax.plot([x[i] - 0.08, x[i] + 0.08], [hi, hi], color="black", lw=0.8)
                hi_txt[i] = max(hi_txt[i], hi)
        for j, (b, v, ht) in enumerate(zip(bars, vals, hi_txt)):
            yy = max(v * 1.20, ht * 1.12)     # 加粗数字并上移，避免贴柱顶
            ax.text(b.get_x() + b.get_width() / 2, yy, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.4, fontweight="bold",
                    color=STAR_RED if j == CROSS else "#222222")
        ax.axhline(1.0, color="#555555", lw=0.9, ls="--", zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(0.4, 30)
        ax.set_yticks([0.5, 1, 2, 5, 10, 20])
        ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"], fontsize=7.5)
        # log 轴惯例：只标十进制主刻度；次刻度保留刻度线但不出标签
        # （否则 matplotlib 在跨度小的 log 轴上会给 2×10^1、3×10^2 之类的次刻度也打标签，
        #   挤在主刻度旁，是排版检查里最常被点的小毛病）
        ax.get_yaxis().set_minor_formatter(mpl_ticker.NullFormatter())
        ax.grid(True, which="major", axis="y", ls=":", alpha=0.4)
        ax.set_ylabel("Ratio (A / B)", fontsize=8)
        title = f"({letters[k]}) {lab} ratio — log scale"
        if key == "latency":
            title += " · 16-thread i7-14650HX"   # 台架条件进 (c) 标题
        ax.set_title(title, fontsize=8.8, loc="left")
    # x 轴单行全名（第 4 根带星号），底部留白给呼吸感
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels([f"{nm} *" if i == CROSS else nm
                             for i, nm in enumerate(x_names)], fontsize=7.8)
    axs[-1].set_xlabel("model pair (A / B)", fontsize=8.2)
    # 图例集中在 (a) 右上空白：族色两块 + Parity 虚线
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(facecolor=CNN_BLUE, edgecolor="#222222", lw=0.5,
                                 label="A = YOLO family (bars 1–2)"),
                           Patch(facecolor=TRANS_ORANGE, edgecolor="#222222", lw=0.5,
                                 label="A = RT-DETR (bars 3–4)"),
                           Line2D([], [], color="#555555", ls="--", lw=0.9,
                                  label="Parity (ratio = 1)")],
                  fontsize=6.4, loc="upper right", framealpha=0.92, ncol=1)
    # 图注：手动断成短行，每行都小于画布宽（自动换行去掉后要防水平截断）
    fig.text(0.5, 0.006,
             "Bars = metric of the first-listed model (A) divided by the second (B);\n"
             "bar colour = family of A: blue = YOLO, orange = RT-DETR;  * marks the cross-family pair.\n"
             "Latency (c): cross-round median protocol, median-of-medians (3 rounds × 240\n"
             "samples/round), 16-thread Intel i7-14650HX; YOLO e2e incl. Python NMS / RT-DETR\n"
             "decoding-inclusive without NMS;\n"
             "whiskers span the cross-round min–max ratio. (a)(b) are intrinsic single values — no whiskers.\n"
             # 分组键（类③）：只标哪几根柱属哪一组，不带数字、不下结论。
             # 2026-09-11 用户拍板删去原末 3 行的结论与 −36%/−13%：带数字的结论在改稿中最易与正文脱钩，
             # 而它在 §4.3 已有；图里重复正文已有的数字收益为零、维护成本为正。
             "Bars 1–3: in-family scaling. Bar 4: cross-family.",
             ha="center", va="bottom", fontsize=6.2, color="#333333")
    fig.subplots_adjust(bottom=0.32, top=0.965, left=0.115, right=0.98)
    return _save(fig, out)


def plot_p2(canon_csv: str | Path, out: Path) -> Path:
    """P2 v3：Pareto 图。v3 改动（2026-09-07 用户拍板 + 审稿意见）：
    - X 轴改 **log**：左端 YOLO11n/v8n（差 2ms）不再压成一团，星标与灰色空心可分；
    - 点旁直接标参数量数字（如 YOLOv8m · 25.9M），点大小仍 ∝ √参数量（图例注明），
      解决"√ 压缩参数差距"的观感问题；
    - 底部补硬件/口径脚注（单窗口 3 轮中位、16 线程 i7-14650HX、NMS 口径、无误差棒原因）。
    v4（2026-09-11 用户拍板）：删掉被支配灰点与虚线前沿。§3.4 的双条件规则下存活边为
    0，七点全非支配，前沿即全集；此时按延迟升序连线会在 YOLOv8l 处向下拐，看上去像
    画错。图注改为"无模型被支配，图为三目标前沿的二维投影"。"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rows = load_plot_canonical(canon_csv)
    # 2026-09-11：不再画被支配灰点，也不再连虚线前沿。
    # §3.4 的双条件支配规则下没有任何模型被支配（存活边 0 条），前沿即全集；
    # 此时若仍按延迟升序连一条线，经过 YOLOv8l 时会向下拐，读者第一眼看到的是
    # "图画错了"。第三轴由标记面积承载，图注写明交集为全集。
    # 标注偏移字典（points，文本中心相对数据点）：log-x 下逐点手摆防文字压点/互叠
    off = {"YOLO11n": (0, 22), "YOLOv8n": (10, -16), "YOLOv8s": (8, -16),
           "YOLOv8m": (-28, 22), "RT-DETR-l": (30, -24),
           "YOLOv8l": (-8, 16), "RT-DETR-x": (0, 18)}
    fig, ax = plt.subplots(figsize=(7.0, 5.1))
    for r in rows:
        s = 9 * np.sqrt(r["params_m"])
        ax.scatter(r["lat_ms"], r["map"], s=s, color=_fcolor(r["name"]),
                   edgecolors="white", linewidths=0.5, zorder=4)
    # LAE top-1 高亮：红边白底星（名字 + 参数量并入同一句红字，省一处标签）
    top1 = max(rows, key=lambda r: lae_score(r["map"], r["lat_ms"], r["params_m"],
                                             DEFAULT_ALPHA, DEFAULT_BETA))["name"]
    tr = next(r for r in rows if r["name"] == top1)
    ax.plot(tr["lat_ms"], tr["map"], "*", ms=24, mfc="white", mec=STAR_RED,
            mew=1.6, zorder=6)
    dx0, dy0 = off[top1]
    ax.annotate(f"LAE top-1 ({top1} · {tr['params_m']:.1f}M)",
                (tr["lat_ms"], tr["map"]), textcoords="offset points",
                xytext=(dx0, dy0), ha="center", va="center", fontsize=8,
                color=STAR_RED, fontweight="bold", zorder=7)
    # 其余点名：名 + 参数量数字（v4 起无被支配点，全部同色，不再区分灰/彩）
    for r in rows:
        if r["name"] == top1:
            continue
        dx, dy = off[r["name"]]
        ax.annotate(f"{r['name']} · {r['params_m']:.1f}M",
                    (r["lat_ms"], r["map"]), textcoords="offset points",
                    xytext=(dx, dy), ha="center", va="center", fontsize=7.4,
                    color="#333333", zorder=5)
    # 轴（log x）与刻度
    ax.set_xscale("log")
    ax.set_xlim(24, 480)
    ax.set_ylim(0.355, 0.575)
    ax.set_xticks([30, 50, 100, 200, 400])
    ax.get_xaxis().set_major_formatter(mpl_ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl_ticker.NullFormatter())   # 只留次刻度线，不出标签
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("End-to-end CPU latency (ms, log) · 16-thread i7-14650HX", fontsize=8.5)
    ax.set_ylabel("Unified COCOeval mAP@50-95 (COCO val2017)", fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    # 图例：族色 / LAE top-1 / 点面积语义（v4 起删去"被支配灰点"与"前沿虚线"两项）
    handles = [
        Patch(facecolor=CNN_BLUE, label="CNN (YOLO family)"),
        Patch(facecolor=TRANS_ORANGE, label="Transformer (RT-DETR)"),
        Line2D([], [], marker="*", ms=16, mfc="white", mec=STAR_RED, mew=1.4,
               linestyle="None", label="LAE top-1"),
    ]
    ax.legend(handles=handles, fontsize=6.6, ncol=2, loc="upper left",
              title="marker area ∝ sqrt(parameter count)\n(exact params labelled beside each model)",
              title_fontsize=6.5, framealpha=0.95)
    # 底部脚注：口径 + 星标澄清（回应"LAE top-1 是否在前沿"）；手动断行防水平截断
    # v4（2026-09-11）：星标行末补"未与第 2 名分离"半句 + n 对配对延迟 CI（原句只说
    # "the fastest model in the set"，把点估计当成了已分离的事实，与 §4.1/§4.3 的
    # "point-estimate ordering" 措辞冲突）。
    fig.text(0.5, 0.02,
             "Single-window sequential protocol: median of 3 rounds (240 samples/round) in one\n"
             "continuous window, so no error bars are drawn; YOLO end-to-end latency includes\n"
             "Python NMS; RT-DETR is decoding-inclusive without NMS.\n"
             "No model is Pareto-dominated under the two-condition rule of Section 3.4, so all\n"
             "seven points belong to the three-objective frontier. The red star marks the LAE\n"
             "top-1 (YOLO11n · 2.7M), which is not separated from the runner-up YOLOv8n: under\n"
             "round-matched pairing over 11 interleaved rounds the latency ratio is 1.052 with\n"
             "a 95% interval of [0.989, 1.181], which does not exclude parity.",
             ha="center", va="bottom", fontsize=6.3, color="#555555")
    fig.subplots_adjust(bottom=0.28, top=0.95, left=0.10, right=0.97)
    return _save(fig, out)


def plot_p3(quant_csv: str | Path, quant_sel_csv: str | Path, base_csv: str | Path,
            int8_csv: str | Path, int8_dir: str | Path, out: Path) -> Path:
    """P3 v4 三面板。v3（2026-09-07 审图意见）：Y 轴加余量（三角不贴轴）；失败原因补工具链措辞
    "in ORT v1.28.0 toolkit"（避免被读成 calibration 写错）；(b) 脚注补 FP32 绝对 mAP 区间；
    ylabel 不再用 U+00F7 字形。

    v4（2026-09-11 用户拍板）：加面板 (c)，把 §4.4 最硬的那个对照画出来 —— 同样的权重、
    同样的算术、同样的 ΔmAP，只因出口格式不同，延迟比就落在 parity 的两侧。柱高 =
    **轮内配对**的延迟比（同一轮 FP32 ÷ 同一轮变体），柱内标该格子的 ΔmAP。
    两个硬约束（用户原话"不满足就别画"）：
      1. 图注必须写"仅 YOLOv8n / YOLOv8s"，因为 m/l 没跑 QOperator、RT-DETR 没有可用的 QOp，
         不写读者会把两条柱外推到全部七个模型；
      2. 比值只能来自交错摘要（轮内配对），**不能**拿 int8_latency_window.csv 里不同窗口的
         FP32 基线去除 —— 那正是正文 §4.4 开头在解释的 8% 窗口效应。
    """
    vol = volume_fractions(quant_csv, quant_sel_csv, quant_csv)
    deltas = mAP_deltas(base_csv, int8_csv)
    models_vol = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "RT-DETR-l", "RT-DETR-x"]
    models_del = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "RT-DETR-l"]
    recipes = ["dynamic", "naive", "selective"]

    int8_dir = Path(int8_dir)
    ratios: dict[str, dict[str, dict[str, float]]] = {}
    for m in INT8_MODELS:
        ratios.update(int8_round_matched(int8_dir / f"interleaved_int8_{m.lower()}_summary.csv"))
    cells = int8_panel_c(ratios, int8_delta_map(int8_dir / "int8_bootstrap_ci.csv"))

    fig = plt.figure(figsize=(8.6, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 0.82])
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1])
    a3 = fig.add_subplot(gs[1, :])
    # ---------- (a) 体积相对 FP32 ----------
    x = np.arange(len(models_vol)); w = 0.27
    for j, rc in enumerate(recipes):
        for i, m in enumerate(models_vol):
            v = vol.get(m, {}).get(rc)
            if v is None:
                continue
            b = a1.bar(x[i] + (j - 1) * w, v, w * 0.92, color=_fcolor(m),
                       hatch=RECIPE_HATCH[rc], edgecolor="#333333", lw=0.3)
            a1.text(b[0].get_x() + b[0].get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=5.8, rotation=90)
    a1.axhline(1.0, color="black", lw=0.8, ls=":")
    # RT-DETR 两档 naive-static 崩溃：灰 ✕ + 原因（不画柱，尺寸虽可测但推理不可用）
    for i, m in enumerate(models_vol):
        if m.startswith("RT-DETR"):
            a1.plot(x[i] + (0 - 1) * w, 0.62, marker="x", color=DOM_GRAY, ms=6, mew=1.4)
    a1.text(4.35, 0.72, "naive-static: QDQ numerical failure\nin ORT v1.28.0 toolkit\n"
                        "(scale=0/NaN), not usable",
            fontsize=5.6, ha="right", color="#666666")
    # ha="right"：默认左对齐会把这段文字推出 xlim（x 轴只到 5.5），压到 (b) 的轴上去
    a1.text(5.45, 1.05, "FP32 = 1", fontsize=6.8, color="#333333", ha="right")
    a1.set_xticks(x); a1.set_xticklabels(models_vol, rotation=20, fontsize=7)
    a1.set_ylabel("Size ratio (INT8 / FP32)", fontsize=8.5)
    a1.set_ylim(0, 1.18)
    a1.set_title("(a) Model size after INT8 (relative to FP32)", fontsize=9)
    from matplotlib.patches import Patch as _P
    a1.legend(handles=[_P(facecolor=CNN_BLUE, label="CNN (YOLO)"),
                       _P(facecolor=TRANS_ORANGE, label="Transformer (RT-DETR)")],
              fontsize=6.2, loc="upper left", framealpha=0.9)

    # ---------- (b) mAPΔ ----------
    x2 = np.arange(len(models_del))
    for d in deltas:
        xi = models_del.index(d["model"])
        a2.scatter(xi, d["delta"], marker=RECIPE_MARKER[d["recipe"]], s=34,
                   color=_fcolor(d["model"]), edgecolors="white", lw=0.5, zorder=4)
        # 逐点标真值：正上方(带符号)。消除"把刻度当点"的可能。
        a2.annotate(f"{d['delta']:+.3f}", (xi, d["delta"]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=5.8, color="#333333")
    # RT-DETR-l naive-static 崩溃 ✕ + 原因（含工具链措辞）
    a2.plot(4, -0.045, marker="x", ms=8, mew=1.6, color=DOM_GRAY, zorder=5)
    a2.text(3.55, -0.0455, "naive-static on RT-DETR:\nQDQ numerical failure in\n"
                           "ORT v1.28.0 toolkit\n(scale=0/NaN), mAP N/A",
            fontsize=5.8, ha="right", color="#666666")
    a2.axhline(0.0, color="black", lw=0.8)
    a2.axhspan(-0.01, 0.01, color="#009E73", alpha=0.12)
    # ±0.01 标签的绿由 #009E73 改深到 #00694D：前者对白底只有 3.42:1，低于 MVA 的 4.5:1 文字对比度要求。
    # 保留绿系是为了与 axhspan 的绿色带保持语义关联；#00694D 对白底 6.71:1，且与色带同色相。
    a2.text(3.35, 0.018, "±0.01", fontsize=6, ha="right", color="#00694D")
    a2.set_xticks(x2); a2.set_xticklabels(models_del, fontsize=7)
    a2.set_ylabel("ΔmAP@50-95 (INT8 − FP32)", fontsize=8.5)
    a2.set_ylim(-0.135, 0.045)
    a2.set_title("(b) Accuracy change, identical 500-image subset", fontsize=9)
    a2.grid(True, ls=":", alpha=0.3)
    for rc in recipes:
        a2.scatter([], [], marker=RECIPE_MARKER[rc], s=30, color="#666666",
                   label=RECIPE_LABEL[rc])
    # 配方图例放面板外右侧空白带，避免压到数据点
    a2.legend(fontsize=6.6, loc="center left", bbox_to_anchor=(1.02, 0.5),
              framealpha=0.9, title="INT8 recipe", title_fontsize=6.8)
    # ---------- (c) 出口格式 × 配方：同样的 ΔmAP，parity 两侧的延迟 ----------
    fmt_color = {"QDQ": "#8FB8D9", "QOperator": CNN_BLUE}
    xg = np.arange(len(INT8_MODELS))
    wc = 0.19
    for i, m in enumerate(INT8_MODELS):
        for j, c in enumerate([c for c in cells if c["model"] == m]):
            xp = xg[i] + (j - 1.5) * wc
            a3.bar(xp, c["ratio"], wc * 0.86, color=fmt_color[c["format"]],
                   hatch=RECIPE_HATCH[c["recipe"]], edgecolor="#333333", lw=0.35, zorder=3)
            # 轮间范围（同一比值的 min/max，逐轮算出来的），不是跨窗口拼的误差棒
            a3.errorbar(xp, c["ratio"],
                        yerr=[[max(c["ratio"] - c["lo"], 0.0)], [max(c["hi"] - c["ratio"], 0.0)]],
                        fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2.0, zorder=4)
            # 柱内标 ΔmAP：同一格子的精度代价。白底小框保证压在纹理上也读得清。
            a3.text(xp, c["ratio"] * 0.5, f"{c['delta']:+.3f}", ha="center", va="center",
                    fontsize=6.2, color="#1A1A1A", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=0.78))
    a3.axhline(1.0, color="#111111", lw=1.2, ls="--", zorder=2)
    # 用轴坐标系定位（x = 轴宽比例），不写数据坐标：数据坐标会随 xlim 自动缩放漂出轴外
    # 靠左放：左边那两根 QDQ 柱只有 0.42/0.47 高，parity 上方是空的；
    # 靠右放会被 1.92 和 1.37 的 QOperator 柱直接穿过去。
    a3.text(0.005, 1.03, "parity = FP32 baseline",
            transform=a3.get_yaxis_transform(), fontsize=6.4, ha="left", va="bottom",
            color="#111111")
    a3.set_xticks([xg[i] + (j - 1.5) * wc
                   for i in range(len(INT8_MODELS)) for j in range(len(INT8_CELLS))])
    a3.set_xticklabels([INT8_CELL_LABEL[c[0]] for _ in INT8_MODELS for c in INT8_CELLS],
                       fontsize=6.2)
    for i, m in enumerate(INT8_MODELS):
        a3.text(xg[i], -0.235, m, ha="center", va="top", fontsize=8.5,
                transform=a3.get_xaxis_transform())
        if i:
            a3.axvline(xg[i] - 0.5, color="#CCCCCC", lw=0.7, ls=":", zorder=1)
    a3.set_ylabel("Latency ratio\n(FP32 / INT8), above 1 = faster", fontsize=8.2)
    a3.set_ylim(0, 2.15)
    a3.set_title("(c) Tested export paths show different INT8 latency", fontsize=9)
    from matplotlib.patches import Patch as _P2
    a3.legend(handles=[_P2(facecolor=fmt_color["QDQ"], edgecolor="#333333", lw=0.35,
                           label="QDQ export"),
                       _P2(facecolor=fmt_color["QOperator"], edgecolor="#333333", lw=0.35,
                           label="QOperator export"),
                       _P2(facecolor="#E8E8E8", edgecolor="#333333", lw=0.35, hatch="//",
                           label="naive static (head in INT8)"),
                       _P2(facecolor="#E8E8E8", edgecolor="#333333", lw=0.35, hatch="xx",
                           label="head-preserving static")],
              fontsize=6.2, loc="upper left", ncol=2, framealpha=0.9,
              title="fill = export format, hatch = recipe", title_fontsize=6.2)

    # 图注：手动断行防水平截断（v4.1 同因：单段超长文字被画布右缘裁掉）
    fig.text(0.5, 0.008,
             "mAPΔ = single evaluation on one identical 500-image COCO subset (no repeats), so no error bars.\n"
             "FP32 subset baselines run 0.396 (YOLOv8n) to 0.539 (YOLOv8l; RT-DETR-l 0.536), so a negative\n"
             "Δ means INT8 sits below FP32. Naive-static Δ = -0.069 (YOLOv8n) and ≈ -0.085/-0.087/-0.087\n"
             "(v8s/m/l); selective static stays inside the ±0.01 band (max |Δ| = 0.0093 on YOLOv8l).\n"
             "RT-DETR static failed in the ONNX Runtime v1.28.0 QDQ toolkit (scale=0/NaN), not on YOLO.\n"
             "In (c) the bar height is the round-matched ratio (each round's FP32 median over the same\n"
             "round's variant median) and the whisker is the spread over the five rounds; the number inside\n"
             "each bar is that cell's mAPΔ. Only YOLOv8n and YOLOv8s appear: the QOperator format was not\n"
             "exported for YOLOv8m/l and no usable QOperator export exists for RT-DETR. Dynamic INT8 is\n"
             "absent from (c) because its cost made an interleaved run impractical; its ratios are\n"
             "same-window figures reported in the text.",
             ha="center", va="bottom", fontsize=6.1, color="#333333")
    fig.subplots_adjust(bottom=0.27, top=0.95, left=0.075, right=0.78,
                        wspace=0.30, hspace=0.34)
    return _save(fig, out)


def plot_p4(canon_csv: str | Path, out: Path) -> Path:
    """P4 v5 双面板（2026-09-07 用户拍板改题，替代被否证的"LAE vs 朴素指标 R²"）：
    (a) LAE α×β 秩稳健热图 —— 宽区 α∈[0.05,1.0]×β∈[0.05,0.6] 步长 0.05 = 240 点，网格在代码内用
        lae_sweep.grid_stability 现算（与账本 §5.1 同源同函数，非手抄）；绿 = 7 模型全排序与默认
        (0.5,0.3) 恒同，粉 = 翻转（实测只出现在 α≤0.15∧β≤0.1 退化角 4 点）；虚线框 = 计划区
        α∈[0.2,0.8]×β∈[0.1,0.5]（117/117 恒同）；红星 = 默认指数。
    (b) 效率余量失真 —— 两对 (YOLOv8n/YOLOv8l、YOLOv8l/RT-DETR-l) 的 E=mAP/FLOPs 与 E=mAP/latency
        余量柱对，log 轴 + parity=1。数字现场从 canonical CSV 重算（v8n/v8l 13.3 vs 6.1、
        v8l/RT-l 0.64 vs 0.83，账本 §5.3），图注用 f-string 拼、不与数据脱节。
    v5.1（2026-09-07 用户反馈）:parity=1 虚线改深色(#111111) lw1.3 zorder3.6 置顶并纳入图例
    （"Parity (margin = 1)"，图例移右上空白角）；图例标签改语义 Theoretical—mAP/GFLOPs /
    Measured CPU—mAP/CPU ms；图注 (b) 开头点破色码：magenta=理论效率(FLOPs)、blue=实测 CPU
    效率(measured ms)，并点明虚线含义（above=第一个模型更高效，below=第二个更高效）。"""

    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D as _L2D
    from matplotlib.patches import Patch, Rectangle

    rows = load_plot_canonical(canon_csv)

    # ---- (a) 秩稳健热图（宽区 240 点）----
    alphas = np.round(np.arange(0.05, 1.001, 0.05), 2).tolist()
    betas = np.round(np.arange(0.05, 0.601, 0.05), 2).tolist()
    stab = grid_stability(rows, alphas, betas)
    n_flip = stab["n_cells"] - stab["n_identity"]
    ident = {(round(c["alpha"], 2), round(c["beta"], 2)): bool(c["identical"])
             for c in stab["cells"]}
    Z = np.array([[ident[(a, b)] for a in alphas] for b in betas], dtype=int)  # 行=β
    # 计划区（虚线框内容，117 点）——同样现算，供图注写命中数
    pl_al = np.round(np.arange(0.2, 0.801, 0.05), 2).tolist()
    pl_be = np.round(np.arange(0.1, 0.501, 0.05), 2).tolist()
    stab_plan = grid_stability(rows, pl_al, pl_be)
    half = 0.025  # 半格宽（步长 0.05），框要罩住整列格心

    # ---- (b) 余量失真（账本 §5.3 口径：ratio = E(第一个模型)/E(第二个)）----
    b_pairs = [("YOLOv8n", "YOLOv8l"), ("YOLOv8l", "RT-DETR-l")]
    b_lab = ["YOLOv8n / YOLOv8l", "YOLOv8l / RT-DETR-l"]
    b_f, b_t = [], []
    for A, B in b_pairs:
        ra = next(r for r in rows if r["name"] == A)
        rb = next(r for r in rows if r["name"] == B)
        b_f.append((ra["map"] / ra["gflops"]) / (rb["map"] / rb["gflops"]))
        b_t.append((ra["map"] / ra["lat_ms"]) / (rb["map"] / rb["lat_ms"]))

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(8.4, 4.5), width_ratios=[1.75, 1.0],
        gridspec_kw={"wspace": 0.42})
    # ---- (a) 面板 ----
    cmap = ListedColormap(["#F2C9C6", "#C9E7C7"])  # 0=翻转(粉) 1=恒同(绿)
    axA.imshow(Z, extent=[alphas[0] - half, alphas[-1] + half,
                          betas[0] - half, betas[-1] + half],
               origin="lower", aspect="equal", interpolation="nearest",
               cmap=cmap, vmin=0, vmax=1)
    axA.add_patch(Rectangle((0.2 - half, 0.1 - half), 0.6 + half, 0.4 + half,
                            fill=False, edgecolor="#444444", lw=1.0, ls="--"))
    axA.plot(DEFAULT_ALPHA, DEFAULT_BETA, marker="*", ms=14, mfc="white",
             mec=STAR_RED, mew=1.6, zorder=6)
    axA.set_xlim(alphas[0] - half - 0.02, alphas[-1] + half + 0.02)
    axA.set_ylim(betas[0] - half - 0.02, betas[-1] + half + 0.02)
    axA.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axA.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    axA.tick_params(labelsize=7.6)
    axA.set_xlabel("Latency exponent  α", fontsize=8.4)
    axA.set_ylabel("Params exponent  β", fontsize=8.4)
    axA.set_title("(a) LAE rank stability on the α × β grid", fontsize=9, loc="left")
    axA.text(0.24, 0.585, f"identical ranking", fontsize=5.8, color="#2F6B31")
    axA.text(0.06, 0.585, f"flip (α,β→0)", fontsize=5.8, color="#8A2B26", ha="left")
    axA.text(DEFAULT_ALPHA + 0.03, DEFAULT_BETA + 0.01, "default (0.5, 0.3)",
             fontsize=6.4, color=STAR_RED, va="center")
    # ---- (b) 面板 ----
    xb = np.arange(2); w = 0.32
    bars_f = axB.bar(xb - w / 2, b_f, w, color=METRIC_COLORS["FLOPs"],
                     edgecolor="#222222", lw=0.5, zorder=3)
    bars_t = axB.bar(xb + w / 2, b_t, w, color=METRIC_COLORS["Latency"],
                     edgecolor="#222222", lw=0.5, zorder=3)
    par_line = axB.axhline(1.0, color="#111111", lw=1.3, ls="--", zorder=3.6,
                           label="Parity (margin = 1)")
    for bars in (bars_f, bars_t):
        for b in bars:
            v = b.get_height()
            axB.text(b.get_x() + b.get_width() / 2, v * 1.14, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=7.0)
    axB.set_yscale("log")
    axB.set_ylim(0.3, 40)
    axB.set_yticks([0.5, 1, 2, 5, 10, 20])
    axB.set_yticklabels(["0.5", "1", "2", "5", "10", "20"], fontsize=7.4)
    axB.get_yaxis().set_minor_formatter(mpl_ticker.NullFormatter())   # 同 P1：次刻度不出标签
    axB.set_xticks(xb)
    axB.set_xticklabels(b_lab, fontsize=7.2)
    axB.set_ylabel("Efficiency margin (A / B) — log", fontsize=8.2)
    axB.set_title("(b) FLOPs vs measured latency", fontsize=9, loc="left")
    axB.legend(handles=[
        Patch(facecolor=METRIC_COLORS["FLOPs"],
              label="Theoretical — mAP / GFLOPs"),
        Patch(facecolor=METRIC_COLORS["Latency"],
              label="Measured CPU — mAP / CPU ms"),
        par_line],
        fontsize=6.2, loc="upper right", framealpha=0.9)
    axB.grid(True, axis="y", ls=":", alpha=0.4)

    # ---- 底部图注（手动短行，v4.2 纪律；数值全用 f-string 现算）----
    cap = "\n".join([
        f"(a) LAE = mAP / (Latency{SUP_ALPHA} × Params{SUP_BETA}), 7 detectors ranked per grid cell (step 0.05): "
        f"green = ranking identical to",
        f"the default (α = 0.5, β = 0.3, red star). The dashed box is the planned region "
        f"α ∈ [0.2, 0.8], β ∈ [0.1, 0.5]:",
        f"{stab_plan['n_identity']}/{stab_plan['n_cells']} identical. The full {stab['n_cells']}-cell grid "
        f"breaks only in the {n_flip}-cell corner where both exponents are",
        "small (α ≤ 0.15, β ≤ 0.1) — there LAE approaches a pure-accuracy order (cost penalty ≈ 0), "
        "the definitional limit, not instability.",
        "(b) Efficiency margin of the first vs the second model of each pair (A / B). Bar color: ",
        "magenta = theoretical efficiency (mAP / GFLOPs); blue = measured CPU efficiency ",
        "(mAP / measured CPU ms). The dashed line is parity (margin = 1): above it the first model ",
        "is more efficient, below it the second. FLOPs overstates the CPU gap — YOLOv8n/YOLOv8l ",
        f"reads {b_f[0]:.1f}× by FLOPs vs {b_t[0]:.1f}× by measured CPU (≈{b_f[0] / b_t[0]:.1f}× ",
        f"overstatement); YOLOv8l/RT-DETR-l reads {b_f[1]:.2f}× vs {b_t[1]:.2f}×, i.e. as ",
        f"1/margin {1 / b_f[1]:.2f}× vs {1 / b_t[1]:.2f}× — measured CPU shows the smaller ",
        f"RT-DETR-l advantage. Latency = the single-window column of Table 2, the column that feeds ",
        f"LAE, on the same 7-model set as the Pareto figure.",
    ])
    fig.text(0.5, 0.012, cap, ha="center", va="bottom", fontsize=6.0, color="#333333")
    fig.subplots_adjust(bottom=0.40, top=0.94, left=0.075, right=0.97)
    return _save(fig, out)


def plot_p5(pairs_csv: str | Path, fit_csv: str | Path, out: Path) -> Path:
    """P5 = 21 对 FLOPs–延迟标度律（log-log 散点 + parity 线 + 拟合线）。S3 新增。

    数据单源：scaling_law.py 落盘的 scaling_law_pairs.csv（21 行，方向按 FLOPs 归一）
    与 scaling_law_fit.csv（all21/e2e 的 γ、截距、CI、R²）。图内数字全部现算，不手抄。
    编码：颜色 = 分子所在族（蓝 YOLO / 橙 RT-DETR，沿用全文族色），点形 = 对类型
    （实心圆 = 同族对，三角 = 跨族对）——跨族三角整体位于拟合线上方是本图要传达的第二条信息。
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch as _P

    pr = _read(pairs_csv)
    fit = {(r["fit"], r["latency"]): r for r in _read(fit_csv)}
    f = fit[("all21", "e2e")]
    gamma, c = float(f["gamma"]), float(f["intercept"])
    lo, hi = float(f["boot_ci_lo"]), float(f["boot_ci_hi"])
    r2 = float(f["r2"])

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    xmax = max(float(p["flops_ratio"]) for p in pr)
    xs = np.logspace(0.0, float(np.log10(xmax)) + 0.03, 200)
    ax.plot(xs, xs, "--", color="#555555", lw=1.0, zorder=2,
            label="Parity (latency ratio = FLOPs ratio)")
    ax.plot(xs, np.exp(c) * xs ** gamma, "-", color="#111111", lw=1.5, zorder=3,
            label=f"OLS fit (γ = {gamma:.2f})")

    for p in pr:
        cross = str(p["cross_family"]).strip().lower() in ("true", "1")
        ax.scatter(float(p["flops_ratio"]), float(p["e2e_ratio"]),
                   s=38, marker="^" if cross else "o", color=_fcolor(p["num"]),
                   edgecolors="white", linewidths=0.6, zorder=4)

    # 只标两对"锚点"（正文手挑的那两对），避免 21 个标签糊成一团。
    # 位置按 21 点的实际分布挑空白带：v8l/v8n 上方无点；RT-x/RT-l 下方 y<1.1 整条带无点。
    for num, den, tx, ty in [("YOLOv8l", "YOLOv8n", 13.0, 13.5),
                             ("RT-DETR-x", "RT-DETR-l", 3.6, 0.70)]:
        row = next(p for p in pr if p["num"] == num and p["den"] == den)
        ax.annotate(f"{num} / {den}", (float(row["flops_ratio"]), float(row["e2e_ratio"])),
                    xytext=(tx, ty), ha="center", va="center", fontsize=7.2, color="#333333",
                    zorder=6, arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.get_xaxis().set_minor_formatter(mpl_ticker.NullFormatter())   # log-log：两轴都只标主刻度
    ax.get_yaxis().set_minor_formatter(mpl_ticker.NullFormatter())
    ax.set_xlim(1.0, xmax * 1.35)
    ax.set_ylim(0.5, max(float(p["e2e_ratio"]) for p in pr) * 2.4)
    ax.set_xlabel("FLOPs ratio (A / B, both axes on a log scale)", fontsize=8.6)
    ax.set_ylabel("Measured CPU latency ratio (A / B)", fontsize=8.6)
    ax.set_title("FLOPs-to-latency scaling across all 21 model pairs", fontsize=9.2, loc="left")
    ax.grid(True, which="both", ls=":", alpha=0.35, zorder=0)

    over = {r: r ** (1.0 - gamma) for r in (2.130, 18.887)}

    # 跨族 / 同族：各自相对"全体拟合线"的几何平均偏移之比（= 图注里的 fitted offset）。
    # 原先图注那个 1.2× 是**手抄字符串**，与本函数 docstring「图内数字全部现算，不手抄」矛盾，
    # 2026-09-12 改为现算。口径：每点偏移 = 实测延迟比 ÷ 拟合线在该 R 处的取值 c·R^γ；
    # 跨族 10 点取几何平均、同族 11 点取几何平均，两者相除 = 1.199 → 仍印 "1.2"，图面文字不变。
    _off = {}
    for _kind in ("cross", "within"):
        _v = [float(p["e2e_ratio"]) / (np.exp(c) * float(p["flops_ratio"]) ** gamma) for p in pr
              if (str(p["cross_family"]).strip().lower() in ("true", "1")) == (_kind == "cross")]
        _off[_kind] = float(np.exp(np.mean(np.log(_v))))
    off_ratio = _off["cross"] / _off["within"]
    ax.text(0.035, 0.955,
            f"γ = {gamma:.2f}  95% CI [{lo:.2f}, {hi:.2f}] (cluster bootstrap)\n"
            f"R² = {r2:.3f},  n = 21 pairs,  7 models\n"
            f"FLOPs overstatement = R{SUP_1MGAMMA}:  {over[2.130]:.2f}× at R = 2.1,  "
            f"{over[18.887]:.2f}× at R = 18.9",
            transform=ax.transAxes, fontsize=6.6, va="top", ha="left", color="#222222",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BBBBBB", lw=0.6),
            zorder=7)

    handles = [
        _P(facecolor=CNN_BLUE, edgecolor="white", label="CNN numerator (YOLO)"),
        _P(facecolor=TRANS_ORANGE, edgecolor="white", label="Transformer numerator (RT-DETR)"),
        Line2D([], [], marker="o", ls="none", markerfacecolor="#888888", markeredgecolor="white",
               ms=6, label="within-family pair"),
        Line2D([], [], marker="^", ls="none", markerfacecolor="#888888", markeredgecolor="white",
               ms=7, label="cross-family pair"),
    ]
    ax.legend(handles=handles, fontsize=6.4, loc="lower right", framealpha=0.95,
              title="point colour / shape", title_fontsize=6.4)

    r_lo = min(float(p["flops_ratio"]) for p in pr)
    r_hi = max(float(p["flops_ratio"]) for p in pr)
    cap = "\n".join([
        "All C(7,2) = 21 unordered pairs of the seven-model set, each oriented so that the numerator has the larger "
        "FLOPs",
        "count, fitted by OLS on logs, from the single-window sequential protocol (the same latency column as the "
        "Pareto figure).",
        f"Because γ < 1, a FLOPs ratio R translates into a latency ratio of only about R{SUP_GAMMA}, "
        "so FLOPs overstate the "
        "measured CPU",
        f"speed advantage by R{SUP_1MGAMMA}: {r_lo ** (1 - gamma):.2f}× at the closest pair (R = {r_lo:.1f}), rising to "
        f"{r_hi ** (1 - gamma):.2f}× at the widest (R = {r_hi:.1f}). The two anchor",
        f"pairs of the efficiency figure read {over[2.130]:.2f}× (R = 2.13) and {over[18.887]:.2f}× (R = 18.9). "
        "Triangles (cross-family pairs) sit above the",
        "fitted line at equal FLOPs ratio on average: a FLOPs-matched CNN-versus-Transformer comparison understates the "
        "Transformer's CPU",
        f"cost (fitted offset ≈ {off_ratio:.1f}×). The 21 pairs share models and are therefore not independent; the interval "
        "quoted is a cluster",
        "bootstrap over the seven models (B = 1000, resampling models and rebuilding the pair set), not a pair-level "
        "bootstrap.",
    ])
    fig.text(0.5, 0.012, cap, ha="center", va="bottom", fontsize=5.9, color="#333333")
    fig.subplots_adjust(bottom=0.30, top=0.93, left=0.085, right=0.975)
    return _save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FIG_DIR))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    r = ROOT / "results"
    p1 = plot_p1(r / "latency_robust.csv", out / "Fig1")
    p2 = plot_p2(r / "latency_canonical7.csv", out / "Fig2")
    p3 = plot_p3(str(r / "int8_quant.csv"), str(r / "int8_quant_sel.csv"),
                 str(r / "map_subset.csv"), str(r / "map_subset_int8.csv"),
                 r, out / "Fig4")
    p4 = plot_p4(r / "latency_canonical7.csv", out / "Fig3")
    p5 = plot_p5(r / "scaling_law_pairs.csv", r / "scaling_law_fit.csv",
                 out / "Fig5")
    print("figures written:")
    for f in (p1, p2, p3, p4, p5):
        print(f"  {f}  ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
