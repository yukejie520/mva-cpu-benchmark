from pathlib import Path
import shutil
p = Path(__file__).resolve().parents[1] / "paper" / "submission"
src = p / "supplementary.pdf"
dst = p / "ESM_1.pdf"
if src.exists():
    shutil.copy2(src, dst)
    print(dst)
