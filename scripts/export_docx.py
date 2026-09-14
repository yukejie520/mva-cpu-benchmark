r"""把组装稿导出成 MVA 投稿要的**可编辑源文件**（方案 A：正文 + SI 两个 docx），并做导出后校验。

为什么必须是"一份 markdown 源 + 一个可声明的转换动作"（2026-09-11 用户拍板）：
    ❌ 手工在 docx 里删水平线 → 每次重新转换都要重来一遍，且一定会漏。
    ❌ 另存一份"导出用 md"     → 两份 md 会漂移，违背本项目的**单一真源**原则。
    ✅ pandoc Lua filter      → 源稿保持唯一真源；转换可声明、可复现、可回滚；pandoc 自带 Lua。

导出后的自动校验（用户要求"和标题白名单放进同一个校验脚本"，这样横线没去掉和假标题
又出现都能被自动拦住）：
    ① 水平线数为 **0**；
    ② 标题结构等于**该文件的白名单**；
    ③ 数字串多重集逐项相等（md ↔ docx）——静默改数字是这条路上最坏的一类错；
    ④ 特殊符号计数相等（× / − / –）；
    ⑤ 无 U+FFFD 乱码；
    ⑥ figures.md 里声明的**每一条**图注都在正文；
    ⑦ 拆分无损：正文 + SI = 全文；
    ⑧ 表格列数（结构）；
    ⑨ 声明段在正文且在位、且不泄漏进 SI；
    ⑩ **回环词级**：docx → markdown，与源稿做词级比对，两个方向都必须为空。

⚠️ ⑩ 为什么必须存在（2026-09-12，kjyu 质疑检查③的抽数方式后补上）：
    ①–⑨ 里唯一能看见"文字有没有变形"的是 ③，而 ③ 只比**数字串**。
    粘字粘的是**空格，不是数字**：源稿 `QDQ<br>naive` 转成 docx 是 `QDQnaive`，
    数字一个没变，③ 的多重集**完全相等**，静默通过。同类还有"整词消失"——
    源稿 `(Processor Information\Processor Frequency)` 里的 `\P` 被 pandoc 当转义符，
    交付的 docx 里 **"Processor" 这个词整个不见了**，同样一个数字都没变。
    这两个缺陷都是 kjyu 要求做回环比对时才被查出来的，且**在 Word 里肉眼可见**。
    → 教训与 §7.16/§7.19 同源：**一个检测器看不见某类缺陷，不等于那类缺陷不存在。**

⚠️ 一个检测器盲区，留痕（2026-09-11 本脚本开发时踩到）：
    第一版水平线检测只找 `<w:pBdr>` 和 `<w:bottom>`，在 pandoc 3.9 上**恒为 0**，
    于是差点得出"pandoc 本来就丢掉 `---`，filter 是多余的"这个**错误结论**。
    真相：pandoc 把水平线编码成 **VML 矩形**（`<v:rect ... o:hr="t"/>`），
    段落数对照（10 vs 9）与回环转换都证明 filter **确实在干活**。
    → 又一次同一个道理：**"没抓到" ≠ "没有"**；正式启用前先用正对照验一遍检测器本身。

用法：
    python scripts/export_docx.py                      # 写 paper/submission/
    python scripts/export_docx.py --out <目录>
"""
from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pypandoc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assemble_draft as ad  # noqa: E402
from lint_headings import token  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LUA = Path(__file__).resolve().parent / "drop_hr.lua"
CONTEXT = ROOT / "paper" / "submission" / "cover_letter.md"
# 组装稿路径：导出口与 assemble_draft.py 写的是**同一个文件**（且经同一个写入口）
ASSEMBLED = ROOT / "paper" / "assembly" / "assembled_draft_v3.md"

SI_HEAD = "# Supplementary Material"
BODY_DROP = {SI_HEAD}   # 这一节要单独成文件，正文里不留

# 水平线在 docx 里的真实编码（pandoc 3.9 实测）：VML 矩形 + o:hr="t"
_HR_SIGNS = ('o:hr="t"', "<w:pBdr", "<w:bottom")

_T = re.compile(r"<w:t[^>]*>([^<]*)</w:t>")
_P = re.compile(r"<w:p[ >].*?</w:p>", re.S)
_STYLE = re.compile(r'<w:pStyle w:val="(Heading[1-9])"')
# Markdown 有序列表的行首编号：转 docx 后变成 Word 自动编号、不进文本层，
# 不剔除就会在数字串 diff 里每轮误报（§7.15 的假告警）。
_MD_LIST_MARK = re.compile(r"(?m)^\s*\d+\.\s+")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def expected_fig_ids() -> list[str]:
    """figures.md 声明了哪几张图（按编号排序）。

    作为校验 ⑥ 的期望集：锚定**行首的图注定义** `**Fig. N.**`，所以图注正文里
    "as in Fig. 2" 这类互引不会被误算进来。手抄常量会随图数增加而腐化（2026-09-12 实测：
    加 Fig. 5 后旧常量只会报「应含 Fig. 1–4」，不告诉你原因），故改为现读。
    """
    text = (ad.MANU / "figures.md").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"(?m)^\*\*Fig\. (\d)\.\*\*", text)))


def check_cover_letter(title: str) -> list[str]:
    """Check the local cover letter before producing final submission files."""
    if not CONTEXT.is_file():
        return [f"缺少投稿 cover letter：{CONTEXT}"]
    text = CONTEXT.read_text(encoding="utf-8")
    problems: list[str] = []
    match = re.search(r'manuscript titled "([^"]+)"', text)
    if not match:
        problems.append("cover letter 未找到 manuscript titled \"...\" 标题句")
    elif match.group(1) != title:
        problems.append(
            f"cover letter 标题与摘要标题不一致：{match.group(1)!r} != {title!r}"
        )
    for marker in ("First,", "Second,", "Third,"):
        if text.count(marker) != 1:
            problems.append(f"cover letter 的 {marker!r} 应恰好出现 1 次")
    return problems


def sections(text: str) -> list[str]:
    """按**一级标题**切段（`## ` 不会命中，因为需要 `# ` 后跟空格）。"""
    return [p for p in re.split(r"(?m)^(?=# )", text) if p.strip()]


def split_body_si(text: str) -> tuple[str, str]:
    """拆成 (正文, SI)。SI = `# Supplementary Material` 那一节；其余按原顺序拼回正文。

    用**同一个分隔符常量** `ad.SEP` 拼回，保证正文与源稿结构一致，
    水平线交给 Lua filter 统一去掉——不在这一步偷偷改结构。
    """
    secs = sections(text)
    si = [s for s in secs if s.startswith(SI_HEAD + "\n") or s.strip() == SI_HEAD]
    if len(si) != 1:
        raise ValueError(f"期望恰好 1 个 SI 节，实际 {len(si)} 个")
    body = [s for s in secs if not (s.startswith(SI_HEAD + "\n") or s.strip() == SI_HEAD)]
    if not body:
        raise ValueError("正文为空")
    return ad.SEP.join(s.strip("\n") for s in body) + "\n", si[0].strip("\n") + "\n"


def hr_count(docx: Path) -> int:
    """docx 里水平线的条数（数 VML 矩形的 o:hr="t"，以及边框写法，双保险）。"""
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
    return sum(xml.count(s) for s in _HR_SIGNS)


def docx_headings(docx: Path) -> list[tuple[int, str]]:
    """从 docx 里读标题：[(层级, 文本)]，按出现顺序。

    走的是文档里**实际写入的段落样式**（Heading1/Heading2），不是我们自己以为写了什么。
    """
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
    out: list[tuple[int, str]] = []
    for para in _P.findall(xml):
        m = _STYLE.search(para)
        if not m:
            continue
        text = "".join(_T.findall(para)).strip()
        out.append((int(m.group(1)[-1]), text))
    return out


def docx_text(docx: Path) -> str:
    """docx 的纯文本（按段落换行），供数字串/符号比对。

    ⚠️ 必须 `html.unescape`：XML 里 `&` 写作 `&amp;`，不解码就会在文本层留下字面量 `&amp;`
    （实测 10 处），使"这段文字在源稿里找得到吗"这类反查误报。
    实测本文档**没有** `&#...;` 数字字符引用，所以不存在"实体凭空造数字"的风险（已核查）。
    """
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
    return "\n".join(html.unescape("".join(_T.findall(p))) for p in _P.findall(xml))


# 归一化：docx 是 zip，其中 docProps/core.xml 每次导出都写入当前时间，
# 于是**裸 md5 每次都变**（与 §7.13 的 PDF 完全同类）。实测两次导出只有这一个成员不同，
# 且只有 dcterms:created / dcterms:modified 两个字段的内容不同，抹掉后逐字节相同。
_TS = re.compile(rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*")


def docx_fingerprint(docx: Path) -> str:
    """去掉时间戳后的指纹 —— **这才是 docx 的有效变更检测器**，裸 md5 不是。"""
    h = hashlib.md5()
    with zipfile.ZipFile(docx) as z:
        for name in sorted(z.namelist()):
            data = z.read(name)
            if name == "docProps/core.xml":
                data = _TS.sub(rb"\1TS", data)
            h.update(name.encode())
            h.update(data)
    return h.hexdigest()


_TBL = re.compile(r"<w:tbl>.*?</w:tbl>", re.S)
_ROW = re.compile(r"<w:tr[ >].*?</w:tr>", re.S)
_CELL = re.compile(r"<w:tc>|<w:tc ")


def docx_tables(docx: Path) -> list[int]:
    """每张表的**列数**（按首行单元格数）。数值本身由数字串多重集比对覆盖，这里管结构。"""
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
    out: list[int] = []
    for tbl in _TBL.findall(xml):
        rows = _ROW.findall(tbl)
        out.append(len(_CELL.findall(rows[0])) if rows else 0)
    return out


_RT_DROP = re.compile(r"[|>`*_#]")
_RT_HYPHENS = re.compile(r"-{2,}")


def roundtrip_md(docx: Path) -> str:
    """docx → markdown（交给 pandoc 自己的解析器）。

    这不是"我们以为写了什么"，而是"Word 里实际是什么"——与 docx_headings 同一个道理。
    """
    return pypandoc.convert_file(str(docx), "markdown", format="docx")


def norm_words(text: str) -> Counter:
    """归一化到可比口径，剔除 pandoc 的**转义产物**（它们不是内容差异）。

    剔除项与理由：
      · `<...>` 原始 HTML → 空格：源稿若写 `<br>`，丢掉的正是那个换行，留个空格保住词边界；
      · `–` → `--`：pandoc 的 markdown writer 把 en dash 转义成两个连字符；
      · 删 `\\`：pandoc 给需转义的字符加反斜杠，反斜杠本身不代表内容；
      · `| > 反引号 * _ #` → 空格：markdown 的表格/引用/代码/强调/标题记号；
      · 连字符串压缩、纯连字符串丢弃：表格边框与水平线（水平线已由 Lua filter 去掉）。

    ⚠️ 本函数的一个**已知盲区**，写在这里免得将来被当成"已经全覆盖"：
        删反斜杠是**双向**的，所以"docx 里反斜杠没了、但词还在"这种差异查不出来。
        代价可接受：那只是排版差异，不会让读者读到一个错词或有歧义的句子。
        本文真正踩到的是"反斜杠带走了整个词"，那个方向会被 source_only 抓到。
    """
    s = re.sub(r"<[^>]+>", " ", text)
    s = s.replace("–", "--").replace("\\", "")
    s = _RT_DROP.sub(" ", s)
    out: Counter = Counter()
    for w in re.sub(r"\s+", " ", s).split():
        w = _RT_HYPHENS.sub("--", w)
        if re.fullmatch(r"-+", w):
            continue
        out[w] += 1
    return out


def roundtrip_diff(docx: Path, md_text: str) -> tuple[list[str], list[str]]:
    """返回 (回环多出的词, 源稿多出的词)，两者都应为空。

    **两个方向都要查**，因为它们对应两种不同的故障：
      · 回环多出 = docx 里的词**粘在了一起**（如 `QDQ<br>naive` → `QDQnaive`）；
      · 源稿多出 = docx 里**丢词或被打散**（如 `\\Processor` 被转义符吃掉 → 整词消失）。
    只查一个方向会漏掉另一半。
    """
    src, rt = norm_words(md_text), norm_words(roundtrip_md(docx))
    return sorted(rt - src), sorted(src - rt)


def numbers(text: str) -> Counter:
    """数字串多重集。传入前**必须**先剔除 md 的行首列表编号（用 strip_list_marks）。"""
    return Counter(_NUM.findall(text))


def strip_list_marks(md: str) -> str:
    return _MD_LIST_MARK.sub("", md)


def convert(md_text: str, tmp_md: Path, out_docx: Path) -> Path:
    """markdown → docx，经 drop_hr.lua 丢掉全部水平线。"""
    tmp_md.parent.mkdir(parents=True, exist_ok=True)
    tmp_md.write_text(md_text, encoding="utf-8")
    pypandoc.convert_file(str(tmp_md), "docx", outputfile=str(out_docx),
                          format="markdown", extra_args=["--lua-filter=" + str(LUA),
                                                          "--resource-path=" + str(ROOT),
                                                          "--variable=fontsize=10pt"])
    return out_docx


# 各文件期望的标题白名单（token 化后比对）
BODY_H1 = ["FLOPs Latency and INT8 Export Formats in a Reproducible CPU Benchmark of YOLO and RT-DETR",
           "1", "2", "3", "4", "5", "6",
           "Statements and Declarations", "Figures", "References"]
BODY_H2_NUM = (["Abstract"]
               + [f"3.{i}" for i in range(1, 8)]
               + [f"4.{i}" for i in range(1, 6)]
               + [f"5.{i}" for i in range(1, 5)])
SI_H1 = ["Supplementary Material"]
SI_H2 = ["Table S1", "Table S2", "Table S3"]

# 表格列数（§7.15 探针逐张核过）：结构项，数值本身由③的数字串多重集覆盖
BODY_TABLE_COLS = [5, 7, 7, 5, 6, 6, 6, 5, 4]
SI_TABLE_COLS = [7, 6, 6]

# 声明段：MVA 投稿系统逐项要问，正文缺一不可；SI 里不该出现
DECLARATIONS = ("Funding.", "Competing interests.", "Author contributions.",
                "Data availability.", "Use of AI tools.")


def check_structure(docx: Path, want_h1: list[str], want_h2: list[str]) -> list[str]:
    hs = docx_headings(docx)
    got1 = [token(t) for l, t in hs if l == 1]
    got2 = [token(t) for l, t in hs if l == 2]
    problems: list[str] = []
    for lvl, got, want in ((1, got1, want_h1), (2, got2, want_h2)):
        gc, wc = Counter(got), Counter(want)
        for t in sorted(set(gc) | set(wc)):
            d = gc[t] - wc[t]
            if d > 0:
                problems.append(f"H{lvl} 多出 {t!r}（出现 {gc[t]} 次，应为 {wc[t]} 次）")
            elif d < 0:
                problems.append(f"H{lvl} 缺少 {t!r}（出现 {gc[t]} 次，应为 {wc[t]} 次）")
    deep = sorted({l for l, _ in hs if l > 2})
    if deep:
        problems.append(f"出现 {deep} 级标题（应只有 1–2 级）")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "submission"))
    args = ap.parse_args()
    out = Path(args.out)
    tmp = out / "_tmp"

    text = ad.assemble()
    problems = ad.check(text)
    if problems:
        print("组装自检失败，先修源稿：")
        for p in problems:
            print(f"  - {p}")
        return 1

    # 顺手把组装稿写出来——**走唯一写入口**，导出口绝不自己写。
    # 先前这里只用内存里的 text、不写盘，于是"只改源稿 + 只跑导出"会让磁盘上的组装稿
    # 停在旧版本而 docx 是新的，两个产物静默不一致（2026-09-12 实测踩到，见 §7.20.7）。
    digest = ad.write_assembled(text, ASSEMBLED)
    print(f"[组装稿] {ASSEMBLED.name}  md5 {digest}")

    # 护栏 2 的后半：**跨进程**核对"两个入口等价"。
    # 不是靠约定，是跑一次 assemble_draft.py 单独看它打印什么，再与本进程比对。
    # 真跨进程才测得到"导入路径 / 编码 / 换行"这类只在另一个进程里才现形的东西。
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    solo = subprocess.run(
        [sys.executable, str(Path(ad.__file__)), "--print-md5"],
        capture_output=True, text=True, env=env, check=True).stdout.strip()
    if solo != digest:
        print(f"\n❌ 两入口不等价：导出口算出 {digest}，assemble_draft.py 单独跑是 {solo}")
        return 1
    print(f"[组装稿] 跨进程核对：assemble_draft.py 单独跑 = {solo}  ✅ 两入口等价")

    body_md, si_md = split_body_si(text)
    body = convert(body_md, tmp / "body.md", out / "manuscript.docx")
    si = convert(si_md, tmp / "si.md", out / "supplementary.docx")

    rc = 0
    title = re.sub(r"^#\s+", "", (ad.MANU / "0_abstract.md")
                   .read_text(encoding="utf-8").strip().splitlines()[0]).strip()
    cover_problems = check_cover_letter(title)
    print("\n[cover letter]")
    if cover_problems:
        print("  ❌ " + "；".join(cover_problems))
        rc = 1
    else:
        print("  ✅ 标题一致，First/Second/Third 三项贡献各 1 次")
    print("=== 导出后校验 ===")
    for docx, name, w1, w2 in ((body, "manuscript.docx", BODY_H1, BODY_H2_NUM),
                               (si, "supplementary.docx", SI_H1, SI_H2)):
        n_hr = hr_count(docx)
        struct = check_structure(docx, w1, w2)
        hs = docx_headings(docx)
        print(f"\n[{name}]")
        print(f"  ① 水平线 : {n_hr} 条" + ("  ✅" if n_hr == 0 else "  ❌ 应全部去掉"))
        print(f"  ② 标题   : H1 ×{sum(1 for l, _ in hs if l == 1)}、H2 ×{sum(1 for l, _ in hs if l == 2)}"
              + ("  ✅" if not struct else "  ❌"))
        for p in struct:
            print(f"       - {p}")
        if n_hr or struct:
            rc = 1

    # ③④⑤ 数字串与符号：**逐文件对逐文件**
    # ⚠️ 必须同文件比。第一版拿"全文 md"比"正文 docx"，而 SI 的三张表已被拆走，
    #    于是表里的延迟范围与节点数全被报成"md 独有"——**比较口径错**，不是内容丢。
    print(f"\n[数字串 / 符号]")
    for md_text, docx, name in ((body_md, body, "manuscript"),
                                (si_md, si, "supplementary")):
        md = strip_list_marks(md_text)
        dx = docx_text(docx)
        nmd, ndx = numbers(md), numbers(dx)
        only_md, only_dx = nmd - ndx, ndx - nmd
        print(f"\n  [{name}]")
        print(f"    ③ 数字串 : md {sum(nmd.values())} 个、docx {sum(ndx.values())} 个"
              + ("  ✅ 多重集相等" if not only_md and not only_dx else "  ❌"))
        if only_md or only_dx:
            print(f"         md 独有 {dict(only_md)}")
            print(f"         docx 独有 {dict(only_dx)}")
            rc = 1
        for ch, label in (("×", "乘号"), ("−", "U+2212 减号"), ("–", "en dash")):
            a, b = md.count(ch), dx.count(ch)
            print(f"    ④ {label:12s}: md {a} / docx {b}  " + ("✅" if a == b else "❌"))
            if a != b:
                rc = 1
        bad = "�" in dx
        print(f"    ⑤ 乱码     : {'❌ 发现 U+FFFD' if bad else '✅ 无 U+FFFD'}")
        if bad:
            rc = 1

    # ⑦ 拆分无损：两个文件的份额相加，必须**恰好等于全文**。
    # 这是独立于逐文件比对的第二道保险：逐文件比对只能证明"每个文件的 md 与 docx 一致"，
    # 若某段内容在**拆分那一刻**就掉进了夹缝，两边都没有它，逐文件比对是发现不了的。
    full, parts = numbers(strip_list_marks(text)), numbers(strip_list_marks(body_md)) + numbers(strip_list_marks(si_md))
    if full == parts:
        print(f"\n  ⑦ 拆分无损 : 正文+SI = {sum(parts.values())} 个数字串 = 全文 "
              f"{sum(full.values())} 个  ✅")
    else:
        print(f"\n  ⑦ 拆分无损 : ❌ 正文+SI 与全文对不上")
        print(f"         拆分后多出 {dict(parts - full)}")
        print(f"         拆分后丢失 {dict(full - parts)}")
        rc = 1

    # ⑥ 图注必须在**正文**文件里，而不是图件文件里（MVA 明文要求）。
    # 2026-09-12 起，期望集**从 figures.md 现读**，不再手抄 "1–4"：手抄常量在新增 Fig. 5 时
    # 只会报 "❌ 应含 Fig. 1–4"，不提示"其实是加了一张图"，得靠人回想才改得动。
    # 现在这条检查的是「figures.md 声明的每一条图注是否都进了正文」，图数再变也不用改这里。
    dx_body = docx_text(body)
    caps = sorted(set(re.findall(r"\bFig\. (\d)\.", dx_body)))
    want = expected_fig_ids()
    have_all = caps == want
    missing = sorted(set(want) - set(caps))
    print(f"\n  ⑥ 图注     : 正文 docx 含 Fig. {','.join(caps)}；figures.md 声明 {','.join(want)}"
          + ("  ✅ 每条都在正文" if have_all else f"  ❌ 正文缺 Fig. {','.join(missing)}"))
    if not have_all:
        rc = 1

    # ⑧ 表格：列数是结构，数值已由③覆盖。总数须回到 §7.15 探针记录的 12 张。
    bt, st = docx_tables(body), docx_tables(si)
    ok_tbl = bt == BODY_TABLE_COLS and st == SI_TABLE_COLS
    print(f"  ⑧ 表格     : 正文 {len(bt)} 张 {bt}、SI {len(st)} 张 {st}"
          + ("  ✅ 与 §7.15 探针一致" if ok_tbl else f"  ❌ 期望 正文 {BODY_TABLE_COLS} / SI {SI_TABLE_COLS}"))
    if not ok_tbl:
        rc = 1

    # ⑨ 声明段：必须在**正文**里（SI 不该有）。缺一不可，投稿系统会逐项问。
    #    反向也要查：声明若漏进 SI，等于同一段话投两遍，编辑会退回。
    dx_si = docx_text(si)
    missing = [k for k in DECLARATIONS if k not in dx_body]
    leaked = [k for k in DECLARATIONS if k in dx_si]
    print(f"  ⑨ 声明段   : 正文含 {len(DECLARATIONS) - len(missing)}/{len(DECLARATIONS)} 项"
          + ("、SI 无泄漏  ✅" if not leaked else f"、❌ SI 误含 {leaked}")
          + ("" if not missing else f"  ❌ 缺 {missing}"))
    if missing or leaked:
        rc = 1

    # ⑩ 回环词级：把 docx 转回 markdown，与源稿做**词级**比对。
    #    为什么 ③ 拦不住这类：粘字粘的是**空格，不是数字**——`QDQ<br>naive` 转成 docx 是
    #    `QDQnaive`，数字一个没变，数字串多重集完全相等，③ 永远看不见。
    #    同类还有"整词消失"（`\Processor` 被 pandoc 当转义符吃掉），也是 ③ 的盲区。
    print("\n[回环词级比对]")
    for md_text, docx, name in ((body_md, body, "manuscript"),
                                (si_md, si, "supplementary")):
        extra, missing = roundtrip_diff(docx, md_text)
        ok = not extra and not missing
        print(f"  [{name}] ⑩ 回环词级 : " + ("✅ 与源稿零差异" if ok else "❌"))
        if extra:
            print(f"        docx 粘出/造出 {len(extra)} 个词：{extra}")
        if missing:
            print(f"        docx 丢失/打散 {len(missing)} 个词：{missing}")
        if not ok:
            rc = 1

    # 清掉中间文件，只留交付物
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    print(f"\n交付物：{out}")
    for d in (body, si):
        print(f"  {d.name}  ({d.stat().st_size / 1024:.0f} KB)  "
              f"指纹 {docx_fingerprint(d)}")
    if rc == 0:
        print("\n✅ 全部校验通过")
    else:
        print("\n❌ 有校验未通过，见上")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
