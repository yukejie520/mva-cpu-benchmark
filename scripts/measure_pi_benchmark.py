"""Pi 5 full benchmark for the frozen cross-platform protocol.

This runner is intentionally separate from the main-platform loader.  It
requires the recorded image manifest, uses a rotated model order, records
per-sample/e2e data and platform telemetry, and refuses to overwrite a prior
Pi run unless ``--force`` is explicit.

Default protocol: six named models, 4 intra-op threads, 640 px, batch size 1,
20 warm-ups, 3 rounds, and 8 images x 30 repetitions per model and round.
YOLOv8m is deliberately not in the default set.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_common import IMGSZ, build_session, preprocess  # noqa: E402
from measure import REGISTRY  # noqa: E402
from onnx_metrics import onnx_gflops, onnx_params_m  # noqa: E402
from platform_telemetry import cpu_frequency_mhz, cpu_temperature_c, system_info, utc_now  # noqa: E402
from protocol_io import image_paths_from_manifest  # noqa: E402
from yolo_post import yolov8_nms  # noqa: E402


DEFAULT_MODELS = [
    "YOLO11n", "YOLOv8n", "YOLOv8s", "YOLOv8l", "RT-DETR-l", "RT-DETR-x",
]


def rotation_orders(n: int, rounds: int) -> list[list[int]]:
    """Rotate the model order once per round."""
    return [[(round_index + offset) % n for offset in range(n)]
            for round_index in range(rounds)]


def tail_stats(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("samples must be non-empty")
    values = np.asarray(samples, dtype=np.float64)
    return {
        "n_samples": int(values.size),
        "median_ms": round(float(np.median(values)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "p99_ms": round(float(np.percentile(values, 99)), 3),
        "max_ms": round(float(np.max(values)), 3),
        "mean_ms": round(float(np.mean(values)), 3),
        "std_ms": round(float(np.std(values)), 3),
    }


def measure_one(sess, cfg: dict, inputs: list[np.ndarray], warmup: int,
                reps: int) -> tuple[list[float], list[float]]:
    """Measure forward and deployment-accounted e2e samples for one model."""
    input_name = sess.get_inputs()[0].name
    output_names = [output.name for output in sess.get_outputs()]
    kind = cfg["kind"]
    for _ in range(warmup):
        output = sess.run(output_names, {input_name: inputs[0]})
        if kind == "yolo":
            yolov8_nms(output[0])

    forward, end_to_end = [], []
    for image in inputs:
        for _ in range(reps):
            started = time.perf_counter()
            output = sess.run(output_names, {input_name: image})
            forward_finished = time.perf_counter()
            if kind == "yolo":
                yolov8_nms(output[0])
                finished = time.perf_counter()
            else:
                finished = forward_finished
            forward.append((forward_finished - started) * 1000.0)
            end_to_end.append((finished - started) * 1000.0)
    return end_to_end, forward


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs", default=str(ROOT / "data" / "coco128" / "coco128" /
                                               "images" / "train2017"))
    parser.add_argument("--manifest", default=str(ROOT / "results" / "main_timing_images.txt"),
                        help="required frozen image list shared with the main platform")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="comma-separated subset of the six default models")
    parser.add_argument("--threads", type=int, default=4,
                        help="Pi 5 intra-op threads; the planned protocol uses 4")
    parser.add_argument("--n-imgs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "pi5"))
    parser.add_argument("--force", action="store_true",
                        help="explicitly replace files in an existing output directory")
    args = parser.parse_args()

    if args.threads <= 0 or args.rounds <= 0 or args.reps <= 0 or args.warmup < 0:
        raise SystemExit("threads/rounds/reps must be positive and warmup non-negative")
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not names or len(set(names)) != len(names):
        raise SystemExit("--models must contain unique model names")
    if "YOLOv8m" in names:
        raise SystemExit("YOLOv8m is intentionally excluded from the Pi protocol")
    registry = {cfg["name"]: cfg for cfg in REGISTRY}
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}")

    out_dir = Path(args.out_dir)
    outputs = [out_dir / "pi5_samples.csv", out_dir / "pi5_summary.csv",
               out_dir / "pi5_metadata.json"]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite existing Pi outputs; use a new --out-dir or --force: "
            + ", ".join(str(path) for path in existing)
        )

    image_paths = image_paths_from_manifest(args.manifest, args.imgs, args.n_imgs)
    inputs = [preprocess(str(path), IMGSZ) for path in image_paths]
    configs = [registry[name] for name in names]
    sessions = {}
    for cfg in configs:
        path = ROOT / cfg["onnx"]
        if not path.is_file():
            raise FileNotFoundError(f"required Pi model is missing: {path}")
        sessions[cfg["name"]] = build_session(str(path), threads=args.threads)

    labels = [cfg["name"] for cfg in configs]
    orders = rotation_orders(len(configs), args.rounds)
    print(f"[protocol] platform=Pi5 threads={args.threads} imgsz={IMGSZ} batch=1 "
          f"images={len(inputs)} warmup={args.warmup} rounds={args.rounds} reps={args.reps}")
    print(f"[images] manifest={args.manifest}")
    for index, path in enumerate(image_paths):
        print(f"  {index}: {path.name}")

    samples: list[dict] = []
    run_intervals: list[dict] = []
    per_model_rounds: dict[str, list[list[float]]] = {name: [] for name in labels}
    run_started = utc_now()
    for round_index, order in enumerate(orders, start=1):
        round_started = utc_now()
        print(f"[round {round_index}/{args.rounds}] order={[labels[i] for i in order]}",
              flush=True)
        round_detail = {"round": round_index,
                        "order": [labels[i] for i in order],
                        "started_utc": round_started}
        for position, model_index in enumerate(order):
            cfg = configs[model_index]
            name = cfg["name"]
            model_path = ROOT / cfg["onnx"]
            started_utc = utc_now()
            started_clock = time.perf_counter()
            freq_before = cpu_frequency_mhz()
            temp_before = cpu_temperature_c()
            e2e, forward = measure_one(sessions[name], cfg, inputs, args.warmup, args.reps)
            freq_after = cpu_frequency_mhz()
            temp_after = cpu_temperature_c()
            elapsed_s = time.perf_counter() - started_clock
            finished_utc = utc_now()
            per_model_rounds[name].append(e2e)
            for sample_index, (e2e_ms, forward_ms) in enumerate(zip(e2e, forward)):
                samples.append({
                    "round": round_index, "order": position, "model": name,
                    "file": str(model_path), "image": image_paths[sample_index // args.reps].name,
                    "image_index": sample_index // args.reps,
                    "rep": sample_index % args.reps + 1,
                    "forward_ms": round(forward_ms, 4), "e2e_ms": round(e2e_ms, 4),
                    "freq_before_mhz": round(freq_before, 2) if freq_before is not None else "",
                    "freq_after_mhz": round(freq_after, 2) if freq_after is not None else "",
                    "temp_before_c": round(temp_before, 2) if temp_before is not None else "",
                    "temp_after_c": round(temp_after, 2) if temp_after is not None else "",
                    "started_utc": started_utc, "finished_utc": finished_utc,
                })
            print(f"  {position}: {name} median={st.median(e2e):.3f} ms "
                  f"elapsed={elapsed_s / 60:.1f} min", flush=True)
            run_intervals.append({
                "round": round_index, "order": position, "model": name,
                "file": str(model_path), "started_utc": started_utc,
                "finished_utc": finished_utc, "elapsed_s": round(elapsed_s, 3),
                "freq_before_mhz": freq_before, "freq_after_mhz": freq_after,
                "temp_before_c": temp_before, "temp_after_c": temp_after,
            })
        round_detail["finished_utc"] = utc_now()
        run_intervals.append(round_detail)

    summary_rows = []
    for cfg in configs:
        name = cfg["name"]
        all_e2e = [value for round_samples in per_model_rounds[name]
                   for value in round_samples]
        stats = tail_stats(all_e2e)
        round_medians = [st.median(values) for values in per_model_rounds[name]]
        path = ROOT / cfg["onnx"]
        summary_rows.append({
            "model": name, "family": cfg["family"],
            "params_M": round(onnx_params_m(str(path)), 3),
            "gflops": round(onnx_gflops(str(path), IMGSZ)["gflops"], 3),
            **stats,
            "round_medians_ms": "|".join(f"{value:.3f}" for value in round_medians),
            "median_of_round_medians_ms": round(st.median(round_medians), 3),
        })

    write_csv(out_dir / "pi5_samples.csv", samples)
    write_csv(out_dir / "pi5_summary.csv", summary_rows)
    metadata = {
        "platform_label": "Pi5",
        "system": system_info(),
        "onnxruntime": ort.__version__,
        "python_platform": platform.platform(),
        "image_dir": str(args.imgs), "image_manifest": str(args.manifest),
        "images": [str(path) for path in image_paths],
        "models": [{"name": cfg["name"], "file": str(cfg["onnx"])} for cfg in configs],
        "threads": args.threads, "batch_size": 1, "imgsz": IMGSZ,
        "warmup": args.warmup, "reps": args.reps, "rounds": args.rounds,
        "started_utc": run_started, "finished_utc": utc_now(),
        "round_orders": [{"round": index + 1,
                          "order": [labels[i] for i in order]}
                         for index, order in enumerate(orders)],
        "run_intervals": run_intervals,
        "outputs": [str(path) for path in outputs],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pi5_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"[csv] {out_dir / 'pi5_samples.csv'}")
    print(f"[csv] {out_dir / 'pi5_summary.csv'}")
    print(f"[metadata] {out_dir / 'pi5_metadata.json'}")
    if {row["model"] for row in summary_rows} >= {"RT-DETR-l", "RT-DETR-x"}:
        med = {row["model"]: row["median_of_round_medians_ms"] for row in summary_rows}
        print(f"[anchor] RT-DETR-x / RT-DETR-l = {med['RT-DETR-x'] / med['RT-DETR-l']:.3f}x")


if __name__ == "__main__":
    main()
