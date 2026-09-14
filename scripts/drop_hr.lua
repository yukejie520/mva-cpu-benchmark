--[[
drop_hr.lua —— pandoc 过滤器：丢掉所有 HorizontalRule（Markdown 里的 `---` 分隔线）。

为什么用过滤器，而不是"手工在 docx 里删"或"另存一份导出用 md"（2026-09-11 用户拍板）：
  ❌ 手工在 docx 里删   → 每次重新转换都要重来一遍，且一定会漏。
  ❌ 另存一份"导出用 md" → 两份 md 会漂移，违背本项目的**单一真源**原则。
  ✅ pandoc Lua filter  → 源稿保持唯一真源；转换动作可声明、可复现、可回滚。
     pandoc **自带 Lua 解释器**，不需要额外装 anything。

用法：
    pandoc input.md -o output.docx --lua-filter=scripts/drop_hr.lua

返回空列表 = 删除该元素。返回 nil = 保持原样（这里不这么写，因为我们要全删）。
]]
function HorizontalRule()
  return {}
end
