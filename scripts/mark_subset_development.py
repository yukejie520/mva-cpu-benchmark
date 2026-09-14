from pathlib import Path
f=Path(__file__).resolve().parents[1]/"paper/manuscript/3_methods.md"
s=f.read_text(encoding="utf-8")
needle="The same measurements serve one further purpose, a consistency check on the vendor column of Table 1 (Section 4.1), which does not place them in any absolute role."
s=s.replace(needle,needle+" The 500-image results are development-stage evidence and are not presented as replacements for a complete re-evaluation.")
f.write_text(s,encoding="utf-8")
