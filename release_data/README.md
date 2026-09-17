# Curated benchmark release data

This directory contains the small, publication-facing data package for the
five-round CPU benchmark. It intentionally excludes COCO images, model
weights, temporary logs, and exploratory files.

- `main7_interleaved_raw.csv`: 35 row-level measurements (7 models × 5 rounds).
- `main7_interleaved_summary.csv`: median, range, p95, and p99 latency summary.
- `main7_metadata.json`: protocol, rotated order, timestamps, and relative model names.
- `thread_sensitivity.csv`: 36 measurements for YOLOv8n, YOLOv8l, and RT-DETR-l at 1/4/8/16 threads over three rounds.
- `model_hashes.csv`: SHA-256 hashes of the seven local ONNX model files used for the benchmark.
- `environment.txt`: interpreter, operating system, CPU, runtime, and key package versions.

The current accuracy values in the manuscript remain provisional until the
complete COCO val2017 evaluation is deposited in a later release.
