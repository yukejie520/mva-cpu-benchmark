"""§3.4 Pareto 边判定：把「什么算一条可以声称的支配边」写成一个可复算的纯函数。

为什么要单独一个脚本（2026-09-11 决定）：
    正文原来靠**厂商 mAP** 的严格支配画前沿，RT-DETR-l 把 YOLOv8l 踢出前沿靠的是
    0.001 的厂商差。审阅 R8 打得对：0.001 低于任何可用测量的分辨率。用户拍板的规则
    是**双条件**，两条都必须成立才敢声称一条支配边：

    (i)  符号一致：厂商列与我们自己 500 图同管线自测列，必须把这个模型排在前面。
         若两个参考给出相反方向，精度轴方向不可判定，不声称支配。
    (ii) 正向证据：配对 bootstrap 95% CI 的**下界 >= 0**，即差异确实为正。

    第 (ii) 条的方向是这里最容易写错的地方。**不能**用「没发现反向就保留」——那是在
    接受零假设，有统计背景的审稿人一句话就能拆。要么要求正向证据（本文采用），要么
    写成对偶的「排除被支配者更准」；两者在本数据集上结果一致，但措辞不同。本脚本按
    用户定的「要求正向证据」口径执行。

    规则必须对**全部 21 个无序对**一致适用并报告存活边数。只对能得出想要的结论的那几对
    用规则，就是 ad hoc。

数据的三个来源（全部只读，不新测）：
    results/latency_canonical7.csv    —— 单窗协议的 e2e 中位延迟与参数量
    results/map_bootstrap7.csv        —— 自测 500 图 mAP 与厂商 mAP
    results/map_paired_bootstrap7.csv —— 全部有序对的配对 ΔmAP 与 95% CI

输出：
    results/pareto_edges.csv —— 每对一行，含两个条件各自的真假与最终判定
    stdout 摘要             —— 候选边数、存活边数、前沿集合、被支配集合

用法：
    python scripts/pareto_edges.py
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LAT_CSV = RESULTS / "latency_canonical7.csv"
BOOT_CSV = RESULTS / "map_bootstrap7.csv"
PAIR_CSV = RESULTS / "map_paired_bootstrap7.csv"
OUT_CSV = RESULTS / "pareto_edges.csv"


def edge_verdict(lat_dom: float, lat_sub: float, par_dom: float, par_sub: float,
                 vendor_dom: float, vendor_sub: float,
                 self_dom: float, self_sub: float,
                 ci_lo: float, ci_hi: float) -> dict:
    """判定「dom 支配 sub」这条边是否可以声称。

    参数里的「dom」是候选支配者、「sub」是候选被支配者。三个轴都是**越小越好**
    （延迟、参数量），只有精度越大越好，所以精度差的符号是反的。

    纯函数：不读文件、不打印，便于单元测试（规则错一次，全文的前沿就错一次）。

    返回 dict：
        warrant   —— 前三轴是否给出一条「结构上可能」的边（延迟与参数都严格更省）
        cond_i    —— 两个精度参考是否同向且都指向 dom
        cond_ii   —— 配对 CI 下界是否 >= 0（正向证据）
        assert    —— 最终是否声称支配 = warrant and cond_i and cond_ii
        reason    —— 一句话说明卡在哪一步，便于回填正文
    """
    # 支配的**硬保证**来自延迟和参数：这两个轴是实测值，不需要统计检验。
    # 精度轴只负责「不拖后腿」，所以它的门槛是「要有正向证据」而不是「必须领先很多」。
    warrant = (lat_dom < lat_sub) and (par_dom < par_sub)

    vendor_gap = round(vendor_dom - vendor_sub, 6)
    self_gap = round(self_dom - self_sub, 6)
    cond_i = (vendor_gap > 0) and (self_gap > 0)

    cond_ii = ci_lo >= 0

    # 条件 (i) 有两种失败方式，不能混为一谈：两个参考都指向"被支配者更准"是**结论明确**，
    # 而两个参考符号相反才是"方向不可判定"。早期版本把两者都写成"不同向"，
    # 会让读 CSV 的人以为 20 对都是参考打架，与事实相反。
    if vendor_gap < 0 and self_gap < 0:
        i_reason = (f"两个参考一致地把被支配者排在前面（厂商差 {vendor_gap:+.4f}，"
                    f"自测差 {self_gap:+.4f}）")
    else:
        i_reason = (f"两个参考符号相反（厂商差 {vendor_gap:+.4f}，"
                    f"自测差 {self_gap:+.4f}），精度轴方向不可判定")

    if not warrant:
        edge = False
        reason = "延迟与参数量没有同时严格更省，不构成候选边"
    elif not cond_i:
        edge = False
        reason = i_reason
    elif not cond_ii:
        edge = False
        reason = (f"配对 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] 下界 < 0，"
                  f"没有正向证据")
    else:
        edge = True
        reason = f"三条件齐备（配对 CI [{ci_lo:+.4f}, {ci_hi:+.4f}] 下界 >= 0）"

    return {"warrant": warrant, "vendor_gap": vendor_gap, "self_gap": self_gap,
            "ci_lo": round(float(ci_lo), 4), "ci_hi": round(float(ci_hi), 4),
            "cond_i": cond_i, "cond_ii": cond_ii, "assert": edge, "reason": reason}


def load_latency() -> dict:
    with open(LAT_CSV, newline="", encoding="utf-8") as f:
        return {r["name"]: {"lat": float(r["e2e_median_ms"]),
                            "par": float(r["params_M"])} for r in csv.DictReader(f)}


def load_accuracy() -> dict:
    with open(BOOT_CSV, newline="", encoding="utf-8") as f:
        return {r["model"]: {"self": float(r["self_map"]),
                             "vendor": float(r["vendor_map"])} for r in csv.DictReader(f)}


def load_pairs() -> dict:
    """(A, B) -> 配对 ΔmAP 的 CI。B 列缺失时立刻报错，不静默用错键。"""
    out = {}
    with open(PAIR_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["A"], r["B"])
            if r["B"] in ("", "1000"):
                raise ValueError(
                    f"{PAIR_CSV.name} 的 B 列是 {r['B']!r}，不是模型名。"
                    "这份 CSV 由 map_bootstrap7.py 生成，出现过重复键覆盖的旧版，"
                    "请先重跑那个脚本。")
            out[key] = (float(r["ci_lo"]), float(r["ci_hi"]))
    return out


def main() -> None:
    lat = load_latency()
    acc = load_accuracy()
    pairs = load_pairs()
    models = list(lat.keys())

    rows, asserted = [], []
    for a, b in combinations(models, 2):
        # 候选方向：延迟与参数都更省的那个当支配者；两个方向都不满足就是一条无关对。
        if lat[a]["lat"] < lat[b]["lat"] and lat[a]["par"] < lat[b]["par"]:
            dom, sub = a, b
        elif lat[b]["lat"] < lat[a]["lat"] and lat[b]["par"] < lat[a]["par"]:
            dom, sub = b, a
        else:
            rows.append({"dominator": "", "dominated": "", "lat_dom": "", "lat_sub": "",
                         "par_dom": "", "par_sub": "", "vendor_gap": "", "self_gap": "",
                         "ci_lo": "", "ci_hi": "", "cond_i": "", "cond_ii": "",
                         "edge_asserted": False, "reason": "两个方向都没有同时更省的一侧"})
            continue

        lo, hi = pairs[(dom, sub)]
        v = edge_verdict(lat[dom]["lat"], lat[sub]["lat"], lat[dom]["par"], lat[sub]["par"],
                         acc[dom]["vendor"], acc[sub]["vendor"],
                         acc[dom]["self"], acc[sub]["self"], lo, hi)
        rows.append({"dominator": dom, "dominated": sub,
                     "lat_dom": round(lat[dom]["lat"], 3), "lat_sub": round(lat[sub]["lat"], 3),
                     "par_dom": round(lat[dom]["par"], 3), "par_sub": round(lat[sub]["par"], 3),
                     "vendor_gap": v["vendor_gap"], "self_gap": v["self_gap"],
                     "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
                     "cond_i": v["cond_i"], "cond_ii": v["cond_ii"],
                     "edge_asserted": v["assert"], "reason": v["reason"]})
        if v["assert"]:
            asserted.append((dom, sub))

    n_cand = sum(1 for r in rows if r["dominator"])
    n_i = sum(1 for r in rows if r["dominator"] and r["cond_i"])
    n_ii = sum(1 for r in rows if r["dominator"] and r["cond_ii"])
    asserters = [r for r in rows if r["dominator"] and r["cond_i"] and r["cond_ii"]]
    print("== Pareto 边判定（规则：符号一致 + 配对 CI 下界 >= 0）==")
    print(f"无序对总数 {len(rows)}，其中在延迟与参数两轴构成候选方向的 {n_cand} 条")
    print(f"  通过条件 (i) 符号一致的  {n_i} 条")
    print(f"  通过条件 (ii) 正向证据的 {n_ii} 条")
    print(f"  两条都通过、因而声称支配的 {len(asserters)} 条\n")
    for r in rows:
        if not r["dominator"]:
            continue
        flag = "边 ✓" if r["edge_asserted"] else "边 ✗"
        print(f"[{flag}] {r['dominator']:11s} -> {r['dominated']:11s} "
              f"(i)={str(r['cond_i']):5s} (ii)={str(r['cond_ii']):5s}  {r['reason']}")

    dominated = {s for _d, s in asserted}
    frontier = [m for m in models if m not in dominated]
    print(f"\n被支配模型 {len(dominated)} 个：{sorted(dominated) if dominated else '无'}")
    print(f"前沿 {len(frontier)} 点：{frontier}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] -> {OUT_CSV.name}")


if __name__ == "__main__":
    main()
