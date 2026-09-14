"""图表文字审计：枚举每张图的**全部**文字来源，并对无法解析的来源报警（fail visible）。

为什么需要这个脚本（2026-09-11 立）：
    本轮做"图内文字 ↔ 图注 ↔ 正文"三方对表时，抽取脚本只抓了**内联字符串**，
    而 plot_p4/plot_p5 的图注是 `cap = "\\n".join([...])` 这种**变量承载**的形式，
    于是整块图注被**静默跳过**，导致得出"Fig. 3 没有图内脚注"这个**假结论**。
    「脚本没抓到」≠「图上没有」——本脚本的用途就是把这个区分变成机制：
    任何渲染文字、但静态解析不出来的调用，都会打印 [UNRESOLVED] 警告并以退出码 1 结束，
    而不是悄悄漏掉。

用法：
    python scripts/audit_figure_text.py                  # 审计 scripts/plot_papers.py
    python scripts/audit_figure_text.py <文件> [<文件>…]  # 审计指定文件

输出：
    按 plot_p* 函数分组，逐条列出「行号 / 文字来源调用 / 内容（f-string 表达式位以 {…} 占位）」；
    末尾给覆盖率小结。有无法解析的来源 → 退出码 1。

只读脚本：不改任何文件、不生成任何产物，重复运行输出逐字节相同（幂等）。
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# mathtext 片段：$...$。其中的 ^ 是**合法**的上标语法，不算 ASCII 脱字符违规。
_MATH_SPAN = re.compile(r"\$[^$]*\$")


def bare_caret(s: str) -> bool:
    """判断字符串是否含**裸** `^`（即 mathtext 之外的 ASCII 脱字符）。

    为什么必须排掉 $...$：
        2026-09-11 把图内公式改成 mathtext 真上标后，$^{\\alpha}$ 本身就带 ^。
        若不排除，工具会对刚修好的东西持续误报——审计工具误报一次就没人再信它。
    """
    return "^" in _MATH_SPAN.sub("", s)

# 会渲染可见文字的方法名（这些都算"文字来源"）
TEXT_CALLS = {
    "text", "annotate", "suptitle",
    "set_title", "set_xlabel", "set_ylabel",
    "set_xticklabels", "set_yticklabels",
}
# 通过关键字 label= 渲染文字的构造函数
LABEL_MAKERS = {"Patch", "Line2D", "Rectangle", "FancyArrowPatch"}
# 第一个位置参数不是文字的调用（annotate 的 xy 在前、text 的 xy 在前）
SKIP_FIRST_ARG = {"annotate", "text"}


def _resolve(node: ast.AST, consts: dict[str, str]) -> str | None:
    """把 AST 节点解析成可读字符串；解析不出来返回 None（= 要报警的情况）。

    - Constant 字符串 → 原样
    - f-string（JoinedStr）→ 字面量部分保留，表达式位置填 {…}
    - 相邻字符串字面量拼接（ast 里相邻字面量已被合并，无需特殊处理）
    - Name → 查函数内的赋值表 consts
    - `"sep".join([...])` → 逐元素解析后用 sep 连起来
    - 列表/元组字面量 → 逐元素解析后用 " | " 连起来（tick 标签常见写法）
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.append(v.value)
                continue
            # f-string 的表达式位：若能解析成常量就**原样显示**，否则占位成 {…}。
            # 为什么必须试一把：SUP_ALPHA = r"$^{\alpha}$" 这类常量被 f-string 引用时，
            # 一律占位会让审计看不见它、也看不见常量里真藏着的裸 ^ —— 同一个盲区换个位置。
            inner = getattr(v, "value", None)
            resolved = _resolve(inner, consts) if inner is not None else None
            out.append(resolved if resolved is not None else "{…}")
        return "".join(out)
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        parts = [_resolve(e, consts) for e in node.elts]
        if any(p is None for p in parts):
            return None
        return " | ".join(parts)  # type: ignore[arg-type]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _resolve(node.left, consts), _resolve(node.right, consts)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, ast.Call):
        f = node.func
        # "\n".join([...]) / "".join((...))
        if isinstance(f, ast.Attribute) and f.attr == "join" and node.args:
            sep = _resolve(f.value, consts)
            seq = node.args[0]
            if sep is None:
                return None
            items = seq.elts if isinstance(seq, (ast.List, ast.Tuple)) else []
            parts = [_resolve(e, consts) for e in items]
            if not items or any(p is None for p in parts):
                return None
            return sep.join(parts)  # type: ignore[arg-type]
    return None


def _collect_consts(fn: ast.AST, base: dict[str, str] | None = None) -> dict[str, str]:
    """收集 `名字 = 字符串（或 join 结果）` 赋值，供 Name 解析回溯。

    base 是上层作用域（模块级）的常量表，**必须传**：
        2026-09-11 的缺陷——本函数从空字典起步，于是函数内的
        `cap = "\\n".join([f"...{SUP_ALPHA}..."])` 解析时看不到模块级的 SUP_*，
        产出带 {…} 的次品，**又覆盖掉模块级那个正确的 cap**（同名 key 后者胜）。
        结果：审计输出了"能看见的假图注"，而真正的 mathtext 被掩盖。
    """
    consts: dict[str, str] = dict(base or {})
    # 先收集纯字面量，再收集 join（join 的元素可能是已收集的名字，简单迭代两轮够用）
    for _ in range(2):
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name):
                    v = _resolve(node.value, consts)
                    if v is not None:
                        consts[tgt.id] = v
    return consts


def _iter_text_sites(fn: ast.AST):
    """产出 (行号, 调用名, 文字节点)。文字节点为 None 表示"这个来源解析不出来"。"""
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        if name in TEXT_CALLS:
            args = node.args
            # text(x, y, s) / annotate(s, xy) 的第一个位置参数是坐标，跳过
            text_arg = None
            if name == "annotate":
                text_arg = args[0] if args else None
            elif name == "text":
                text_arg = args[2] if len(args) > 2 else None
            else:
                text_arg = args[0] if args else None
            if text_arg is None and name in SKIP_FIRST_ARG:
                yield node.lineno, name, None
            elif name == "set_xticklabels" or name == "set_yticklabels":
                yield node.lineno, name, args[0] if args else None
            else:
                yield node.lineno, name, text_arg
        elif name in LABEL_MAKERS:
            for kw in node.keywords:
                if kw.arg == "label":
                    yield node.lineno, f"{name}(label=)", kw.value


def audit(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    total = dynamic = 0
    carets: list[tuple[int, str, str]] = []
    # 模块级常量也要进解析表：SUP_ALPHA = r"$^{\alpha}$" 这类常量若被 f-string 引用，
    # 只扫函数体会解析不出来（占位成 {…}），于是常量里真藏了裸 ^ 也看不见——那是个盲区。
    module_consts = _collect_consts(tree)
    if module_consts:
        print(f"[模块级常量] 解析到 {len(module_consts)} 个，纳入名字回溯："
              f"{', '.join(sorted(module_consts))}")
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("plot_p")):
            continue
        consts = _collect_consts(node, module_consts)
        print(f"\n{'=' * 76}\n{node.name}  ({path.name}:{node.lineno})\n{'=' * 76}")
        for lineno, call, text_node in sorted(_iter_text_sites(node)):
            total += 1
            s = _resolve(text_node, consts) if text_node is not None else None
            if s is None:
                dynamic += 1
                print(f"  [DYNAMIC] 行 {lineno}: {call}(...) —— 文字来自运行时值（循环变量 / 列表推导 / "
                      f"未绑定名），静态解析不了，**需人工过目确认它只是数据派生标签、不是成句的图注**")
                continue
            for i, line in enumerate(s.split("\n")):
                tag = f"  [行 {lineno}] {call}: " if i == 0 else " " * 22
                print(f"{tag}{line}")
            if bare_caret(s):
                carets.append((lineno, call, s.strip().split("\n")[0][:60]))
                print("  [⚠️ 裸 ^] ↑ 此行含 mathtext 之外的 ASCII 脱字符。"
                      "图内公式一律用真上标（Latency$^{\\alpha}$），见 2026-09-11 用户拍板。")
    print(f"\n{'-' * 76}")
    print(f"覆盖检查：共发现 {total} 个文字来源；静态解析成功 {total - dynamic} 个，"
          f"数据派生（需人工过目）{dynamic} 个，**静默跳过 0 个**。")
    if carets:
        print(f"\n⚠️  裸 ^ 检查：**{len(carets)} 处**图内文字含 ASCII 脱字符（应全部改成 mathtext 真上标）：")
        for lineno, call, head in carets:
            print(f"    - 行 {lineno}  {call}(...)  {head!r}")
    else:
        print("✅ 裸 ^ 检查：图内文字无 mathtext 之外的 ASCII 脱字符。")
    if dynamic:
        print("⚠️  上面每条 [DYNAMIC] 都要用眼睛确认一遍（这是本脚本存在的意义：宁可多报，不可漏报）。")
        print("    确认要点：它是否只是一串模型名/刻度标签？如果其实是成句的图注，就必须改成可解析的写法。")
        return 1
    if carets:
        return 1
    print("✅ 全部文字来源均已解析（含 cap = \"\\n\".join([...]) 这类变量承载的图注）。")
    return 0


def main() -> int:
    targets = [Path(a) for a in sys.argv[1:]] or [Path(__file__).with_name("plot_papers.py")]
    rc = 0
    for t in targets:
        rc |= audit(t)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
