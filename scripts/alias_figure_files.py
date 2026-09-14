"""Create MVA-friendly Fig1--Fig5 artwork names without deleting legacy files."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
PAIRS = {
    "P1_efficiency_mismatch": "Fig1",
    "P2_pareto": "Fig2",
    "P3_int8_sensitivity": "Fig4",
    "P4_lae_robustness": "Fig3",
    "P5_scaling_law": "Fig5",
}

for old, new in PAIRS.items():
    for ext in (".png", ".pdf"):
        src, dst = FIG / (old + ext), FIG / (new + ext)
        if src.exists():
            shutil.copy2(src, dst)
            print(dst)
