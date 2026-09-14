from pathlib import Path
f=Path(__file__).resolve().parents[1]/"README.md"; s=f.read_text(encoding="utf-8")
s=s.replace("云端实测 fwd 排序与 LAE 降序与本地**完全一致（PASS）** → 跨平台秩一致性证据已入账本 §2 Y 挡小节（`results/kaggle_fwd.csv` 存档）", "该云端结果仅保留为开发记录；修订稿不将其作为跨平台证据（`results/kaggle_fwd.csv` 存档）")
f.write_text(s,encoding="utf-8")
