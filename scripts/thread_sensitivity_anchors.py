"""Run the three-anchor 1/4/8/16-thread sensitivity sweep."""
from pathlib import Path
import argparse, csv, json, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
ANCHORS="YOLOv8n,YOLOv8l,RT-DETR-l"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--imgs",default=str(ROOT/"data/coco128/coco128/images/train2017")); ap.add_argument("--n-imgs",type=int,default=8)
    ap.add_argument("--rounds",type=int,default=3); ap.add_argument("--warmup",type=int,default=20); ap.add_argument("--reps",type=int,default=30)
    ap.add_argument("--out",default=str(ROOT/"results/thread_sensitivity.csv")); a=ap.parse_args()
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); rows=[]
    for t in (1,4,8,16):
        raw=out.with_name(f"_thread_{t}.csv")
        cmd=[sys.executable,str(ROOT/"scripts/measure.py"),"--imgs",a.imgs,"--n-imgs",str(a.n_imgs),"--warmup",str(a.warmup),"--reps",str(a.reps),"--threads",str(t),"--models",ANCHORS,"--out",str(raw)]
        subprocess.run(cmd,check=True)
        with raw.open(encoding="utf-8") as f:
            for row in csv.DictReader(f): row["threads"]=t; rows.append(row)
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(out)
if __name__=="__main__": main()
