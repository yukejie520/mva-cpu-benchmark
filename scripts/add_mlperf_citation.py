from pathlib import Path
f=Path(__file__).resolve().parents[1]/"paper/manuscript/2_related_work.md"; s=f.read_text(encoding="utf-8")
s=s.replace("Rigorous inference suites such as MLPerf [9] established", "Rigorous inference suites such as MLPerf [9,19] established")
f.write_text(s,encoding="utf-8")
