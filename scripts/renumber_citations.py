"""Convert author-year citations in manuscript Markdown to MVA numeric style."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FILES = [p for p in (ROOT / "paper" / "manuscript").glob("*.md") if p.name != "references.md"]
mapping = {
    "Allmendinger et al., 2025": "[1]",
    "Carion et al., 2020": "[2]", "Zhao et al., 2024": "[15]",
    "Dean and Barroso, 2013": "[3]", "Jacob et al., 2018": "[4]",
    "Lin et al., 2014": "[5]", "Ma et al., 2018": "[6]",
    "Reddi et al., 2019": "[9]", "Redmon et al., 2016": "[10]",
    "Suchý and Turčaník, 2026": "[11]", "Williams et al., 2009": "[14]",
    "Ultralytics, 2023": "[12]", "Ultralytics, 2024": "[13]",
    "onnxruntime issue 20052": "[8]", "16009": "[7]",
}
for path in FILES:
    text = path.read_text(encoding="utf-8")
    for old, new in mapping.items():
        text = text.replace(old, new.strip("[]"))
    # Turn citation delimiters into numeric square brackets where the content is numeric.
    text = re.sub(r"\[([^\]\n]*\[(?:\d+)\][^\]\n]*)\]", lambda m: m.group(0), text)
    text = text.replace("[2; [15]]", "[2,15]").replace("[10; [12], [13]]", "[10,12,13]")
    text = re.sub(r"\[\[([0-9, ]+)\]\]", r"[\1]", text)
    path.write_text(text, encoding="utf-8")
