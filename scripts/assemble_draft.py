"""把 paper/manuscript/ 的分节源稿拼成一份可通读的 assembled draft。

为什么要脚本而不是手工粘：
    手工拼过两次（v1 审阅稿、v2 中间稿），两次都漏了后续改动 —— v2 用的还是旧标题
    "Fixed-Exponent Selection Index"、旧的 §3.5 标题、旧的 §4.3 标题，而正文早已改过。
    拼稿是纯机械动作，出错只可能是因为人在做。改成一条命令，并加一道自检：
    拼完立刻检查七个编号小节齐全、标题与 0_abstract.md 一字不差。

幂等性（红线四允许一次性脚本用重跑自检代替测试）：
    同样输入必须产出逐字节相同的输出。脚本自己跑两遍比对 md5，不一致就报错退出。

用法：
    python scripts/assemble_draft.py                 # 写 paper/assembly/assembled_draft_v3.md
    python scripts/assemble_draft.py --out X.md      # 写指定路径
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANU = ROOT / "paper" / "manuscript"

# 允许 `python scripts/assemble_draft.py` 与 `python -m scripts.assemble_draft` 两种跑法
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_headings import lint_text  # noqa: E402

# 拼装顺序：摘要（含标题与作者）→ 六个编号小节 → 声明（back matter）→ 补充材料 → 图 → 参考文献
# 文件名 8 大于 7 却排在 7 前面：声明是正文后的 back matter，而 7_supplementary.md 的
# 文件名被多处引用，改号会牵动一堆引用，所以让 8 插在 6 与 7 之间，顺序由本表说了算。
ORDER = [
    "0_abstract.md",
    "1_introduction.md",
    "2_related_work.md",
    "3_methods.md",
    "4_results.md",
    "5_discussion.md",
    "6_conclusion.md",
    "8_declarations.md",
    "7_supplementary.md",
    "figures.md",
    "references.md",
]

# 分隔符**前后都必须有空行**。原值 "\n---\n\n" 前面只有一个 \n，于是 `---` 紧贴在上一段
# 文字的正下方；而 Markdown 里"一段文字 + 紧接一行 ---" = **setext 二级标题**（`---` 被吃掉）。
# 源稿 0_abstract.md 里关键词后面本来有空行，是组装时被吃掉的，所以问题出在**拼接**这一层。
# 2026-09-11 实测：该缺陷在成稿里制造了 9 个假 H2（含关键词行、若干粗体/斜体起头的段落），
# Word 导航窗格、目录、排版都会跟着错，而**逐字读 markdown 永远看不出来**。
# 修法：加一个前导 \n，让 `---` 成为真正的 thematic break。
SEP = "\n\n---\n\n"


def assemble(manu: Path = MANU) -> str:
    """按 ORDER 拼接；缺文件直接报错，不静默跳过某一节。"""
    missing = [n for n in ORDER if not (manu / n).exists()]
    if missing:
        raise FileNotFoundError(f"分节源稿缺失：{missing}（目录 {manu}）")
    parts = []
    for name in ORDER:
        text = (manu / name).read_text(encoding="utf-8").strip("\n")
        parts.append(text)
    return SEP.join(parts) + "\n"


def check(text: str) -> list[str]:
    """自检：返回问题列表（空 = 通过）。

    只查"拼装"这一层能查的东西：小节在不在、标题与摘要文件是否一致、
    有没有把测试/草稿的痕迹带进来。正文内容对不对不是这里的事。
    """
    problems = []
    for n in range(1, 7):
        if not re.search(rf"^# {n} \S", text, re.MULTILINE):
            problems.append(f"缺少编号小节 '# {n} ...'")
    for head in ("# Supplementary Material", "# Figures", "# References"):
        if head not in text:
            problems.append(f"缺少 '{head}'")
    if "## Abstract" not in text:
        problems.append("缺少 '## Abstract'")

    # 标题必须与 0_abstract.md 首行逐字一致，防止拼到旧标题（v2 就是这么错的）
    src_title = (MANU / "0_abstract.md").read_text(encoding="utf-8").strip().splitlines()[0]
    if src_title not in text:
        problems.append(f"摘要文件的标题未出现在拼装稿里：{src_title!r}")
    if text.strip().splitlines()[0] != src_title:
        problems.append("拼装稿的首行不是标题")

    # 草稿痕迹
    for bad in ("TODO", "FIXME", "<!--"):
        if bad in text:
            problems.append(f"残留草稿痕迹 {bad!r}")

    # 渲染级标题结构自检（2026-09-11 新增，见 lint_headings.py 的模块 docstring）。
    # 拼装这一层最隐蔽的错，是分隔符吃掉了 `---` 前的空行、把一段普通正文顶成 setext 标题：
    # 源文本里一个 `#` 都没有，所以"数井号"这类文本级检查永远看不见。
    # 实测这缺陷在成稿里造出 10 个假 H2。放在这里 = 以后不可能再产出坏稿。
    problems += lint_text(text)
    return problems


def write_assembled(text: str, path: Path) -> str:
    """**唯一写入口**。原子写 + 回读断言 + 指纹留痕 + 陈旧警告，返回写盘后的 md5。

    为什么必须是唯一写入口（2026-09-12 用户拍板方案一）：
        先前有**两条**路径都能产出组装结果——`assemble_draft.py` 写盘、`export_docx.py`
        只用内存里的 `assemble()` 不写。于是"只改源稿 + 只跑导出"会让磁盘上的组装稿
        停在旧版本，而 docx 是新的，**两个产物静默不一致**（实测发生过）。
        设两个入口都调用本函数，写入点就只剩一个，漂移在原理上不可能。

    ⚠️ **护栏 1（用户立）：本函数只写"未经变换的"组装稿。**
        导出期的一切变换——丢 `---`、`^` 上标、反斜杠处理——必须在 **pandoc / Lua 层**做，
        **不许**在 markdown 字符串上动刀。一旦有人在导出路径里对 md 文本做变换，
        "两个入口写出的字节相同"这个前提当场失效，本函数也就失去意义。
        现状符合：丢横线走的是 `scripts/drop_hr.lua`。

    ⚠️ **护栏 2：写完立刻从磁盘读回来逐字节比对，不符即抛错。**
        这一步**不会假阳性**——它比的是"同一个进程刚写下的东西"，不涉及跨进程的
        换行/编码往返。会误报的守卫最后会被当噪音忽略，所以只有不会误报的断言才放在这里。

    ⚠️ `write_bytes` 而非 `write_text`，不能换回去：Windows 上文本模式写盘会把 `\\n`
        翻译成 `\\r\\n`，磁盘 506 行全变 CRLF，于是**磁盘 md5 与内存 md5 变成两个值**
        （实测 `632da908…` vs `88ea7513…`）。那正是"重复生产"的又一个变体。
        写字节 = 磁盘与内存逐字节一致，指纹只有一个含义。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")

    # 陈旧提示：不阻塞，只让"覆盖了旧版本"这件事在输出里可见
    if path.exists():
        old = path.read_bytes()
        if old != data:
            print(f"[WARN] disk md was stale ({hashlib.md5(old).hexdigest()[:8]}), overwritten")

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)          # 原子写：先写临时文件
    tmp.replace(path)              # 再替换，中途失败不会留下半截稿

    back = path.read_bytes()
    if back != data:
        raise RuntimeError(f"回读断言失败：{path} 的磁盘内容与内存结果不一致")
    return hashlib.md5(back).hexdigest()


def assemble_md5(text: str) -> str:
    """组装稿在**内存中**的指纹。磁盘版由 write_assembled 保证与它相等。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "assembly" / "assembled_draft_v3.md"))
    ap.add_argument("--print-md5", action="store_true",
                    help="只算并打印组装稿 md5，不写盘。供导出口跨进程核对两入口等价。")
    args = ap.parse_args()

    text = assemble()
    problems = check(text)
    if problems:
        print("自检失败：")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    # 幂等自检：同样输入跑第二遍，必须逐字节相同
    again = assemble()
    if hashlib.md5(text.encode()).hexdigest() != hashlib.md5(again.encode()).hexdigest():
        print("幂等失败：两次拼装结果不一致（说明拼装引入了随机或时间相关的成分）")
        sys.exit(1)

    if args.print_md5:
        print(assemble_md5(text))
        return

    out = Path(args.out)
    digest = write_assembled(text, out)
    print(f"[ok] {out}")
    print(f"     {len(text.splitlines())} 行 / {len(text.split())} 词 / "
          f"{out.stat().st_size / 1024:.0f} KB")
    print(f"     md5 {digest}")


if __name__ == "__main__":
    main()
