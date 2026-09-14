"""渲染级标题 lint：用 Markdown 解析器把稿子**实际渲染**一遍，断言标题结构等于白名单。

为什么必须是渲染级，而不是扫文本（2026-09-11 立）：

    组装器原先用 `SEP = "\\n---\\n\\n"` 拼接，`---` 前面没有空行，于是它紧贴上一段文字的正下方。
    Markdown 规定"一段文字 + 紧接一行 `---`" = **setext 二级标题**（`---` 被吃掉），
    于是一段普通正文被渲染成了 H2。实测在成稿里制造了 **9 个假标题**，
    而 `#` 计数、正则扫 `^#` 这类**文本级**检查**一个都发现不了**——因为源文本里根本没写 `#`。

    ⚠️ 一个已踩过的坑，值得写进注释：第一版扫描脚本按**行首字符**过滤（跳过以 `*` / `-` 开头的行，
    以为是列表），结果 9 处里漏掉 5 处——全是**粗体或斜体起头**的段落，正好被这条启发式排除。
    → 教训：**任何"按行首字符判断这是不是列表"的启发式，都会系统性漏掉强调起头的段落。**
    交给解析器判断就没有这个盲区：解析器知道什么是列表、什么是段落、什么是 setext 标题。

用法：
    python scripts/lint_headings.py                      # 默认检查 assembled_draft_v3.md
    python scripts/lint_headings.py <文件> [<文件>…]

退出码：0 = 结构完全符合白名单；1 = 有偏差（多、少、重复、或层级不对）。

只读脚本：不改任何文件、不生成产物，重复运行输出逐字节相同（幂等）。
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "paper" / "assembly" / "assembled_draft_v3.md"
TITLE_SRC = ROOT / "paper" / "manuscript" / "0_abstract.md"

# 期望的 H1：标题 + 六个编号小节 + 四个 back matter 节 = 11 个
_NAMED_H1 = ("Statements and Declarations", "Supplementary Material", "Figures", "References")
# 期望的 H2：Abstract + 3.1–3.7 + 4.1–4.5 + 5.1–5.4 + Table S1–S3 = 20 个
_H2_NUMBERED = ([f"3.{i}" for i in range(1, 8)]
                + [f"4.{i}" for i in range(1, 6)]
                + [f"5.{i}" for i in range(1, 5)])
_H2_NAMED = ("Abstract", "Table S1", "Table S2", "Table S3")

_NUM_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)\b")
_TABLE_PREFIX = re.compile(r"^(Table S\d+)")


def token(text: str) -> str:
    """把标题文本压成可比对的**结构标记**，忽略标题措辞的合法改动。

    '3.1 Protocol and ...'      -> '3.1'
    'Table S1. Fusion rate ...' -> 'Table S1'
    'Abstract'                  -> 'Abstract'
    'Statements and Declarations' -> 'Statements and Declarations'
    """
    t = text.strip()
    m = _NUM_PREFIX.match(t)
    if m:
        return m.group(1)
    m = _TABLE_PREFIX.match(t)
    if m:
        return m.group(1)
    return t


def expected_h1() -> list[str]:
    """H1 白名单。标题本身从 0_abstract.md 首行取，避免两处硬编码漂移。"""
    title = TITLE_SRC.read_text(encoding="utf-8").strip().splitlines()[0].lstrip("# ").strip()
    return [title] + [str(i) for i in range(1, 7)] + list(_NAMED_H1)


def expected_h2() -> list[str]:
    return list(_H2_NAMED[:1]) + _H2_NUMBERED + list(_H2_NAMED[1:])


def headings(text: str) -> list[tuple[int, str, int]]:
    """解析并返回 [(层级, 标题原文, 行号)]，**按渲染顺序**。

    行号取 token.map[0]+1（markdown-it 的 map 是 0-based、且 setext 标题的 map 从段首算起）。
    """
    md = MarkdownIt("commonmark")
    tokens = md.parse(text)
    out: list[tuple[int, str, int]] = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            text = inline.content if inline is not None and inline.type == "inline" else ""
            lineno = (tok.map[0] + 1) if tok.map else -1
            out.append((int(tok.tag[1]), text, lineno))
    return out


def compare(got: list[str], want: list[str]) -> list[str]:
    """按**多重集**比对（不是集合）：重复的标题也要报出来。

    ⚠️ 这里第一版写错了，值得留痕：当时写成
        `extra = [t for t in got if t not in want]`
    —— 那是**成员测试**，不是多重集差。后果：一个**白名单里已有的**标题若被复制了一份
    （`## 3.1` 出现两次），它"在 want 里"，于是既不算多、也不算少，**静默通过**。
    而"多出一个标题"恰恰就是本次真实故障的形态，只不过程度不同。
    → 正确做法是 Counter 相减：多几个报几个，少几个也报几个。
    """
    gc, wc = Counter(got), Counter(want)
    problems: list[str] = []
    for t in sorted(set(gc) | set(wc)):
        d = gc[t] - wc[t]
        if d > 0:
            n = f"（出现 {gc[t]} 次，应为 {wc[t]} 次）" if wc[t] else f"（出现 {gc[t]} 次）"
            problems.append(f"多出标题 {t!r}{n}")
        elif d < 0:
            problems.append(f"缺少标题 {t!r}（出现 {gc[t]} 次，应为 {wc[t]} 次）")
    return problems


def lint_text(text: str, verbose: bool = False) -> list[str]:
    """对**文本**做结构检查，返回问题列表（空 = 通过）。

    与 lint() 分开，是为了让组装器能在**写出文件之前**直接对内存里的字符串自检——
    否则就得先落盘再读回来，多一次无谓的 IO，而且"先写坏稿再发现"没有意义。
    """
    hs = headings(text)
    problems: list[str] = []

    deep = [(lvl, txt, ln) for lvl, txt, ln in hs if lvl > 2]
    for lvl, txt, ln in deep:
        problems.append(f"行 {ln}: 出现 {lvl} 级标题 {txt[:60]!r}（本稿只允许 1–2 级）")

    h1 = [t for lvl, t, _ in hs if lvl == 1]
    h2 = [t for lvl, t, _ in hs if lvl == 2]
    problems += compare([token(t) for t in h1], expected_h1())
    problems += compare([token(t) for t in h2], expected_h2())
    return problems


def lint(path: Path, verbose: bool = True) -> list[str]:
    """返回问题列表（空 = 通过）。"""
    hs = headings(path.read_text(encoding="utf-8"))
    problems = lint_text(path.read_text(encoding="utf-8"))

    deep = [(lvl, txt, ln) for lvl, txt, ln in hs if lvl > 2]
    h1 = [t for lvl, t, _ in hs if lvl == 1]
    h2 = [t for lvl, t, _ in hs if lvl == 2]

    if verbose:
        print(f"== {path.name} ==")
        print(f"  解析到 H1 ×{len(h1)}、H2 ×{len(h2)}"
              + (f"、H3+ ×{len(deep)}" if deep else "、无 H3 以下"))
        for lvl, txt, ln in hs:
            if lvl > 2:
                continue
            print(f"    H{lvl}  行 {ln:>4}  {txt[:88]}")
        if not problems:
            print(f"  ✅ 结构等于白名单：H1 ×{len(expected_h1())}、H2 ×{len(expected_h2())}")
    return problems


def main() -> int:
    targets = [Path(a) for a in sys.argv[1:]] or [DEFAULT_TARGET]
    rc = 0
    for t in targets:
        if not t.exists():
            print(f"❌ 找不到文件：{t}")
            rc = 1
            continue
        probs = lint(t)
        if probs:
            rc = 1
            print(f"  ❌ {len(probs)} 处结构偏差：")
            for p in probs:
                print(f"     - {p}")
            print("  提示：若某段正文被渲染成标题，多半是它下面紧跟着一行 `---`（setext 标题）。")
            print("        空行是必须的——`---` 前面没有空行就会被当成标题下划线吃掉。")
        print()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
