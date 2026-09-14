from pathlib import Path
f = Path(__file__).resolve().parents[1] / "paper" / "manuscript" / "5_discussion.md"
s = f.read_text(encoding="utf-8")
s = s.replace("and the export format rather than the integer arithmetic decides whether INT8 is faster here, are the transferable part", "and the QDQ/QOperator latency contrast is toolchain-specific, are the transferable part")
f.write_text(s, encoding="utf-8")
