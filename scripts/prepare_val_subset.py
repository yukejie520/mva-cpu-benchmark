"""准备 COCO val2017 子集：抽取标注 json + 确定性子集选择 + 图片下载。

为什么存在（账本 §3 / notes 计划）：
- 本地要测 FP32 vs INT8 的 mAPΔ，但本机没有 COCO val 标注 → 需下载。
- 全量 5000 张太慢；取 500 张确定性子集（种子固定、全模型共用同一批图），
  AP 与官方口径同源（ultralytics 匹配），INT8 Δ 与 FP32 基线同批图可比。

产物（out_dir 下）：
- instances_val2017.json    全量标注（从官方 zip 抽出的单文件，241MB）
- instances_val500.json     只含选定 500 张 image 的子集标注（images/annotations/categories）
- chosen.csv                选定 image_id,file_name（可复现、留痕）
- images/<file_name>        500 张原图（已存在则跳过）

用法：
    python scripts/prepare_val_subset.py \
        --zip data/coco_val500/annotations_trainval2017.zip \
        --out data/coco_val500 --n 500 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ANN_ZIP_ENTRY = "annotations/instances_val2017.json"  # 官方 zip 内条目带 annotations/ 前缀
IMG_URL = "http://images.cocodataset.org/val2017/{name}"  # 图片直链


# ---------- 纯函数（可脱离网络测试） ----------

def select_images(images: list[dict], n: int, seed: int = 0) -> list[dict]:
    """确定性子集：洗牌取前 n 个后按 file_name 升序（保证全模型同一顺序、可复现）。"""
    if n <= 0 or n > len(images):
        raise ValueError(f"n 必须在 (0, {len(images)}] 之间")
    rng = random.Random(seed)
    pool = list(images)
    rng.shuffle(pool)
    return sorted(pool[:n], key=lambda d: d["file_name"])


def filter_annotations(annotations: list[dict], keep_ids: set[int]) -> list[dict]:
    """只保留 image_id 属于 keep_ids 的标注。"""
    return [a for a in annotations if a["image_id"] in keep_ids]


def build_subset_json(images_all: list[dict], annotations_all: list[dict],
                      categories: list[dict], n: int, seed: int) -> dict:
    """从全量 json 列表生成子集 json（COCO 结构不变）。"""
    chosen = select_images(images_all, n, seed)
    ids = {im["id"] for im in chosen}
    return {
        "images": chosen,
        "annotations": filter_annotations(annotations_all, ids),
        "categories": categories,
    }


def extract_entry(zip_path: str | Path, entry: str, out_path: str | Path) -> Path:
    """从已下载的官方 zip 中抽取单个 json 条目到 out_path（仅一次）。"""
    out = Path(out_path)
    if out.exists():
        return out
    with zipfile.ZipFile(str(zip_path)) as z:
        names = z.namelist()
        if entry not in names:
            raise KeyError(f"zip 里没有 {entry}；现有条目示例：{names[:5]}")
        with z.open(entry) as src, open(out, "wb") as dst:
            dst.write(src.read())
    return out


def _http_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_images(file_names: list[str], dest: str | Path,
                    base_url: str = IMG_URL, n_threads: int = 8,
                    opener=None, timeout: int = 120) -> list[str]:
    """并行下载图片到 dest；已存在跳过。返回下载失败的文件名列表。

    opener 可注入（测试用）：opener(url: str, timeout: int) -> bytes。
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    opener = opener or _http_bytes

    def _one(name: str) -> str | None:
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            return None  # 已下载，跳过
        try:
            target.write_bytes(opener(base_url.format(name=name), timeout))
            return None
        except Exception as e:  # noqa: BLE001 —— 单图失败不中断整批
            print(f"[warn] {name}: {e}", file=sys.stderr)
            return name

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        failed = [f for f in ex.map(_one, file_names) if f is not None]
    return failed


# ---------- 主流程 ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="官方 annotations_trainval2017.zip 路径")
    ap.add_argument("--out", required=True, help="输出目录（如 data/coco_val500）")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-images", action="store_true", help="只出 json，不下载图片")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    full_json = out / "instances_val2017.json"
    print(f"[1/4] 从 zip 抽取 {ANN_ZIP_ENTRY} ...")
    extract_entry(args.zip, ANN_ZIP_ENTRY, full_json)
    coco = json.loads(full_json.read_text(encoding="utf-8"))

    print(f"[2/4] 全量 {len(coco['images'])} 图 -> 确定性子集 n={args.n} seed={args.seed}")
    subset = build_subset_json(coco["images"], coco["annotations"],
                               coco["categories"], args.n, args.seed)
    subset_json = out / "instances_val500.json"
    subset_json.write_text(json.dumps(subset), encoding="utf-8")
    chosen = [(im["id"], im["file_name"]) for im in subset["images"]]
    with open(out / "chosen.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("image_id", "file_name")] + chosen)
    print(f"      子集 json={subset_json.name} 图数={len(chosen)} 标注数={len(subset['annotations'])}")

    if args.no_images:
        return
    names = [im["file_name"] for im in subset["images"]]
    print(f"[3/4] 并行下载 {len(names)} 张图 -> {out / 'images'}")
    failed = download_images(names, out / "images")
    print(f"[4/4] 完成。下载失败 {len(failed)} 张：{failed[:10]}")


if __name__ == "__main__":
    main()
