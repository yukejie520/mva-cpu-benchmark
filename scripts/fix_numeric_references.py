from pathlib import Path
import re
root = Path(__file__).resolve().parents[1] / "paper" / "manuscript"
repl = {
    r"Allmendinger et al\. \[2025\]": "Allmendinger et al. [1]",
    r"Suchý and Turčaník \[2026\]": "Suchý and Turčaník [11]",
    r"\[10; 12, 2024\]": "[10,12]", r"\[12, 2024\]": "[12]",
    r"\[8; 7\]": "[8,7]", r"\[2; 15\]": "[2,15]",
    r"\[15, Table\]": "[15]",
}
for p in root.glob("*.md"):
    if p.name == "references.md": continue
    s = p.read_text(encoding="utf-8")
    for a,b in repl.items(): s = re.sub(a,b,s)
    p.write_text(s, encoding="utf-8")
s = (root / "references.md").read_text(encoding="utf-8")
for i, line in enumerate(s.splitlines(), 1):
    if i > 2 and line.strip() and not line.startswith("["):
        s = s.replace(line, f"[{i-2}] {line}", 1)
(root / "references.md").write_text(s, encoding="utf-8")
