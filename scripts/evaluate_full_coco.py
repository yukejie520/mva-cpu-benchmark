"""Evaluate the seven FP32 checkpoints on a prepared full COCO val2017 set."""
from pathlib import Path
import argparse, csv, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
MODELS=["YOLO11n","YOLOv8n","YOLOv8s","YOLOv8m","YOLOv8l","RT-DETR-l","RT-DETR-x"]
FILES={"YOLO11n":"yolo11n.onnx","YOLOv8n":"yolov8n.onnx","YOLOv8s":"yolov8s.onnx","YOLOv8m":"yolov8m.onnx","YOLOv8l":"yolov8l.onnx","RT-DETR-l":"rtdetr-l.onnx","RT-DETR-x":"rtdetr-x.onnx"}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--images",default=str(ROOT/"data/coco_val2017/val2017")); ap.add_argument("--annotations",default=str(ROOT/"data/coco_val2017/annotations/instances_val2017.json")); ap.add_argument("--out",default=str(ROOT/"results/full_coco_map.csv")); ap.add_argument("--threads",type=int,default=16); a=ap.parse_args()
    img=Path(a.images); ann=Path(a.annotations)
    if not img.exists() or not ann.exists(): raise FileNotFoundError("Prepare full COCO first with scripts/prepare_full_coco_val.py")
    rows=[]
    for name in MODELS:
        model=ROOT/"data"/FILES[name]
        if not model.exists(): raise FileNotFoundError(model)
        kind="yolo" if name.startswith("YOLO") else "rtdetr"
        cmd=[sys.executable,str(ROOT/"scripts/eval_map.py"),"--onnx",str(model),"--name",name,"--kind",kind,"--img-root",str(img),"--anns",str(ann),"--out",str(a.out)]
        subprocess.run(cmd,check=True)
    print(a.out)
if __name__=="__main__": main()
