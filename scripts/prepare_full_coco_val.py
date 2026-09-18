"""Prepare all COCO val2017 images from the official image list.

The script downloads only the official archive, verifies its SHA-256 when
provided, and writes a deterministic manifest. It never commits images or
weights to the repository.
"""
from pathlib import Path
import argparse, hashlib, json, zipfile, urllib.request
ROOT=Path(__file__).resolve().parents[1]
URL="https://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL="https://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def download_atomic(url, destination):
    """Download to a temporary path and publish only after it is complete."""
    destination = Path(destination)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and zipfile.is_zipfile(destination):
        return
    if destination.exists():
        print(f"Removing incomplete archive {destination}")
        destination.unlink()
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, partial)
    if not zipfile.is_zipfile(partial):
        raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {partial}")
    partial.replace(destination)

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=str(ROOT/"data/coco_val2017")); ap.add_argument("--url",default=URL); ap.add_argument("--archive",default=None); ap.add_argument("--expected-sha256",default=None); ap.add_argument("--annotations-url",default=ANNOTATIONS_URL); ap.add_argument("--annotations-source",default=None, help="provenance URL when an annotation mirror is used")
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); archive=Path(a.archive) if a.archive else out/"val2017.zip"
    download_atomic(a.url, archive)
    digest=sha256(archive)
    if a.expected_sha256 and digest.lower()!=a.expected_sha256.lower(): raise ValueError(f"SHA-256 mismatch: {digest}")
    with zipfile.ZipFile(archive) as z: z.extractall(out)
    anns=out/"annotations"/"instances_val2017.json"
    ann_archive=out/"annotations_trainval2017.zip"
    if not anns.exists():
        download_atomic(a.annotations_url, ann_archive)
        with zipfile.ZipFile(ann_archive) as z: z.extract("annotations/instances_val2017.json",out)
    if not anns.exists(): raise RuntimeError("Missing annotations/instances_val2017.json")
    ann_digest=sha256(anns)
    imgdir=out/"val2017"; images=sorted(str(p.relative_to(imgdir)) for p in imgdir.glob("*.jpg"))
    if len(images)!=5000: raise RuntimeError(f"Expected 5000 images, found {len(images)}")
    manifest={"archive":str(archive),"sha256":digest,"count":len(images),"images":images,"annotations":str(anns),"annotations_sha256":ann_digest}
    if a.annotations_source:
        manifest["annotations_source"] = a.annotations_source
    (out/"val2017_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Prepared {len(images)} images; image_sha256={digest}; annotation_sha256={ann_digest}")
if __name__=="__main__": main()
