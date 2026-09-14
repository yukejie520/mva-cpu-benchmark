"""Prepare all COCO val2017 images from the official image list.

The script downloads only the official archive, verifies its SHA-256 when
provided, and writes a deterministic manifest. It never commits images or
weights to the repository.
"""
from pathlib import Path
import argparse, hashlib, json, zipfile, urllib.request
ROOT=Path(__file__).resolve().parents[1]
URL="http://images.cocodataset.org/zips/val2017.zip"
def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=str(ROOT/"data/coco_val2017")); ap.add_argument("--url",default=URL); ap.add_argument("--archive",default=None); ap.add_argument("--expected-sha256",default=None)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); archive=Path(a.archive) if a.archive else out/"val2017.zip"
    if not archive.exists():
        print(f"Downloading {a.url} -> {archive}"); urllib.request.urlretrieve(a.url,archive)
    digest=sha256(archive)
    if a.expected_sha256 and digest.lower()!=a.expected_sha256.lower(): raise ValueError(f"SHA-256 mismatch: {digest}")
    with zipfile.ZipFile(archive) as z: z.extractall(out)
    imgdir=out/"val2017"; images=sorted(str(p.relative_to(imgdir)) for p in imgdir.glob("*.jpg"))
    if len(images)!=5000: raise RuntimeError(f"Expected 5000 images, found {len(images)}")
    (out/"val2017_manifest.json").write_text(json.dumps({"archive":str(archive),"sha256":digest,"count":len(images),"images":images},indent=2)+"\n",encoding="utf-8")
    print(f"Prepared {len(images)} images; sha256={digest}")
if __name__=="__main__": main()
