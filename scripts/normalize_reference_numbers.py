from pathlib import Path
import re
p = Path(__file__).resolve().parents[1] / "paper" / "manuscript" / "references.md"
lines = p.read_text(encoding="utf-8").splitlines()
out = []
n = 0
for line in lines:
    if re.sub(r"^\[\d+\]\s*", "", line).startswith("# References") or not line.strip():
        out.append(line)
        continue
    line = re.sub(r"^\[\d+\]\s*", "", line)
    n += 1
    out.append(f"[{n}] {line}")
p.write_text("\n".join(out) + "\n", encoding="utf-8")
