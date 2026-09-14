"""Five-round interleaved FP32 benchmark for the seven-detector main table.

Writes raw per-round medians, summary statistics (median/range/p95/p99),
round-matched pair ratios, and a JSON manifest. No missing model is silently
omitted: a missing ONNX file raises before the run starts.
"""
from __future__ import annotations
import argparse, csv, json, statistics as st, time
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

REGISTRY = [
    ("YOLO11n", "CNN", ROOT / "data/yolo11n.onnx", "yolo"),
    ("YOLOv8n", "CNN", ROOT / "data/yolov8n.onnx", "yolo"),
    ("YOLOv8s", "CNN", ROOT / "data/yolov8s.onnx", "yolo"),
    ("YOLOv8m", "CNN", ROOT / "data/yolov8m.onnx", "yolo"),
    ("YOLOv8l", "CNN", ROOT / "data/yolov8l.onnx", "yolo"),
    ("RT-DETR-l", "Transformer", ROOT / "data/rtdetr-l.onnx", "rtdetr"),
    ("RT-DETR-x", "Transformer", ROOT / "data/rtdetr-x.onnx", "rtdetr"),
]

def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def rotate(n, rounds): return [[(r+i) % n for i in range(n)] for r in range(rounds)]
def timed(sess, kind, inputs, warmup, reps):
    from yolo_post import yolov8_nms
    name = sess.get_inputs()[0].name; outs = [o.name for o in sess.get_outputs()]
    for _ in range(warmup):
        out = sess.run(outs, {name: inputs[0]})
        if kind == "yolo": yolov8_nms(out[0])
    vals=[]
    for x in inputs:
        for _ in range(reps):
            t0=time.perf_counter(); out=sess.run(outs,{name:x})
            if kind == "yolo": yolov8_nms(out[0])
            vals.append((time.perf_counter()-t0)*1000)
    return vals

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--imgs", default=str(ROOT/"data/coco128/coco128/images/train2017"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=30); ap.add_argument("--n-imgs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--models", default=None, help="comma-separated subset for smoke tests; default all seven")
    ap.add_argument("--raw-out", default=str(ROOT/"results/main7_interleaved_raw.csv"))
    ap.add_argument("--summary-out", default=str(ROOT/"results/main7_interleaved_summary.csv"))
    ap.add_argument("--metadata-out", default=str(ROOT/"results/main7_metadata.json"))
    a=ap.parse_args()
    from eval_common import IMGSZ, build_session, list_images, preprocess
    from platform_telemetry import cpu_frequency_mhz, cpu_temperature_c
    from yolo_post import yolov8_nms
    selected = REGISTRY if not a.models else [r for r in REGISTRY if r[0] in set(a.models.split(','))]
    if not selected: raise ValueError("--models selected no registered model")
    missing=[str(p) for _,_,p,_ in selected if not p.exists()]
    if missing: raise FileNotFoundError("Missing model artifacts: "+", ".join(missing))
    paths=list_images(a.imgs,a.n_imgs)
    if not paths: raise FileNotFoundError(f"No JPG images under {a.imgs}")
    inputs=[preprocess(str(p),IMGSZ) for p in paths]
    sessions={name:build_session(str(path),threads=a.threads) for name,_,path,_ in selected}
    run_started=now(); raw=[]; per={name:[] for name,_,_,_ in selected}; rounds_meta=[]
    for r, order in enumerate(rotate(len(selected),a.rounds),1):
        rounds_meta.append({"round":r,"order":[selected[i][0] for i in order],"started_utc":now()})
        for pos,i in enumerate(order):
            name, family, path, kind=selected[i]; fb=cpu_frequency_mhz(); tb=cpu_temperature_c()
            ts=timed(sessions[name],kind,inputs,a.warmup,a.reps); fa=cpu_frequency_mhz(); ta=cpu_temperature_c()
            row={"round":r,"order":pos,"model":name,"family":family,"kind":kind,"file":str(path),
                 "median_ms":round(float(np.median(ts)),4),"p95_ms":round(float(np.percentile(ts,95)),4),
                 "p99_ms":round(float(np.percentile(ts,99)),4),"max_ms":round(float(max(ts)),4),
                 "n_samples":len(ts),"freq_before_mhz":fb if fb is not None else "","freq_after_mhz":fa if fa is not None else "",
                 "temp_before_c":tb if tb is not None else "","temp_after_c":ta if ta is not None else "","started_utc":now()}
            raw.append(row); per[name].append(row["median_ms"])
        rounds_meta[-1]["finished_utc"]=now()
    base=per[selected[0][0]]; summary=[]
    for name,family,path,kind in selected:
        vals=per[name]; ratios=[b/v for b,v in zip(base,vals)] if name != "YOLO11n" else [1.0]*a.rounds
        summary.append({"model":name,"family":family,"median_ms":round(st.median(vals),4),"range_min_ms":min(vals),"range_max_ms":max(vals),
                        "p95_round_median_ms":round(float(np.percentile(vals,95)),4),"p99_round_median_ms":round(float(np.percentile(vals,99)),4),
                        "ratio_vs_YOLO11n_median":round(st.median(ratios),4),"ratio_min":round(min(ratios),4),"ratio_max":round(max(ratios),4),
                        "round_ratios":"|".join(f"{x:.5f}" for x in ratios),"rounds":a.rounds})
    for out, rows in [(Path(a.raw_out),raw),(Path(a.summary_out),summary)]:
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    meta={"protocol":"five-round interleaved FP32 main benchmark","platform":"Intel Core i7-14650HX / Windows 11","threads":a.threads,
          "imgsz":IMGSZ,"warmup":a.warmup,"reps":a.reps,"n_imgs":len(paths),"images":[str(p) for p in paths],
          "round_orders":rounds_meta,"models":[{"name":n,"family":f,"file":str(p),"kind":k} for n,f,p,k in selected],
          "started_utc":run_started,"finished_utc":now(),"raw_out":a.raw_out,"summary_out":a.summary_out}
    Path(a.metadata_out).write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Wrote {a.raw_out}, {a.summary_out}, {a.metadata_out}")
if __name__ == "__main__": main()
