from pathlib import Path
p = Path(__file__).resolve().parents[1] / "paper" / "manuscript"
for name in ("2_related_work.md", "5_discussion.md"):
    f = p / name; s = f.read_text(encoding="utf-8")
    s = s.replace("What is new is that the export format rather than the integer arithmetic decides speed, with the same weights running slower than FP32 in the QDQ format and 1.11× to 1.92× faster in the QOperator format.", "The observed QDQ/QOperator contrast is toolchain-specific; activation types were not fully matched between all artifacts, so we do not claim that format alone decides speed.")
    s = s.replace("and the export format rather than the integer arithmetic decides whether INT8 is faster here, are the transferable part, because they concern how measurement behaves rather than the value of a measurement.", "are the transferable part, because they concern how measurement behaves rather than the value of a measurement. The QDQ/QOperator latency contrast remains toolchain- and activation-type-specific.")
    f.write_text(s, encoding="utf-8")
