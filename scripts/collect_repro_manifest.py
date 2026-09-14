"""Collect model hashes and environment metadata without copying large assets."""
from pathlib import Path
import argparse, hashlib, json, platform, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
MODELS=["yolo11n.onnx","yolov8n.onnx","yolov8s.onnx","yolov8m.onnx","yolov8l.onnx","rtdetr-l.onnx","rtdetr-x.onnx"]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=str(ROOT/"results/repro_manifest.json")); a=ap.parse_args()
    rows=[]
    for name in MODELS:
        p=ROOT/"data"/name
        rows.append({"file":str(p.relative_to(ROOT)),"exists":p.exists(),"sha256":digest(p) if p.exists() else None,"bytes":p.stat().st_size if p.exists() else None})
    try: pip=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True).splitlines()
    except Exception: pip=[]
    payload={"python":platform.python_version(),"platform":platform.platform(),"machine":platform.machine(),"processor":platform.processor(),"models":rows,"pip_freeze":pip}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); print(out)
if __name__=="__main__": main()
