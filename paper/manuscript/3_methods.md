# 3 Methods

## 3.1 Detectors and weights

We benchmark seven object detectors spanning the two architectural families, all official COCO-pretrained weights used without fine-tuning:

- **CNN family (YOLO-lineage).** YOLO11n [13] and YOLOv8 n / s / m / l [12].
- **Transformer family.** RT-DETR-l / x (HGNetv2 backbone) [15].

All models come from the same source repository (Ultralytics) and were exported to ONNX with the vendor's export pipeline, so that inference runtime, input preprocessing, and weight provenance are identical across families. The set spans roughly one order of magnitude in measured latency (33.0–410.9 ms for the five-round medians), 2.7–66.4 M parameters, and 6.5–225.0 GFLOPs (Table 1). YOLO11n, released after YOLOv8, is included because its position on the accuracy–latency frontier is unknown a priori and our index is evaluated on the final seven-model set.

**Table 1.** Benchmark detectors (official Ultralytics COCO-pretrained weights, 640 px, no fine-tuning).

| Model | Family | Params (M) | FLOPs (G) | Unified full-val mAP50-95 |
|---|---|---|---|---|
| YOLO11n | CNN | 2.66 | 6.54 | 0.387 |
| YOLOv8n | CNN | 3.19 | 8.74 | 0.367 |
| YOLOv8s | CNN | 11.20 | 28.60 | 0.443 |
| YOLOv8m | CNN | 25.93 | 78.94 | 0.495 |
| RT-DETR-l | Transformer | 32.83 | 105.60 | 0.515 |
| YOLOv8l | CNN | 43.71 | 165.15 | 0.521 |
| RT-DETR-x | Transformer | 66.37 | 224.98 | 0.531 |

*FLOPs counted at runtime on the exported ONNX graph (2 × MACs over Conv/ConvTranspose/MatMul/Gemm, so RT-DETR-l is definitionally lower than the vendor's 110.2 G, see §3.2). Unified mAP = the same 5,000-image COCO val2017 evaluation pipeline for all seven models; the external vendor values are retained only as a consistency check.*

## 3.2 Hardware and software configuration

All latency measurements were conducted on one platform: a laptop-class Intel Core i7-14650HX CPU (16 physical cores / 24 threads, 16 GB RAM) running Windows 11, with **turbo boost enabled**, matching how a laptop CPU is deployed in practice. No GPU was used at any point. The "no accelerator" constraint is deliberate, since our claims concern pure-CPU deployment. Inference ran under ONNX Runtime 1.28.0 (CPU execution provider) at **16 intra-op threads** and batch size 1, with 640 px letterboxed COCO input. We chose 16 rather than all 24 intra-op threads after a sweep: throughput improved from 8 to ~16 intra-op threads, while opening all 24 increased end-to-end latency by about 28 % (hyper-thread contention). The CPU supports AVX2 and AVX-VNNI, the 256-bit integer dot-product extension, and does not support AVX-512. Because turbo frequencies drift with load and temperature, frequency before and after each model was logged during the five-round campaign. The primary metadata record spans about 31 min and contains 35 model-round records, with frequency values present for every record; the local temperature interface returned no values and is reported as unavailable. We therefore report the five-round median and its observed range rather than treating any single frequency snapshot as a model-specific causal explanation.

**FLOPs.** Since FLOPs is a central object of this study, we do not take vendor card values but count FLOPs at runtime on the exported ONNX graphs: only Conv / ConvTranspose / MatMul / Gemm operators are counted as 2 × MACs (aligning with the Ultralytics convention), with tensor shapes read from one forward pass. Skip connection operators and normalization/attention-bookkeeping ops are excluded. Counts reproduce vendor values within ≈ 0.5 % for YOLO models. The one discrepancy, RT-DETR-l (105.6 GFLOPs counted vs. 110.2 G reported), is definitional and is reported in Table 1 with the counted value.

## 3.3 Latency measurement protocols

Latency on a laptop CPU is not a stable scalar: absolute timings drift by tens of percent across windows as the scheduler, memory state, and thermal headroom change. Any single round of measurement is therefore a sample of a distribution, not the quantity itself. We report **end-to-end latency** with the deployment-accounting asymmetry made explicit: for YOLO detectors the timed region includes the forward pass plus Python post-processing (decode + non-maximum suppression), because a deployed YOLO cannot emit detections without it. For RT-DETR the forward pass already contains query decoding, and no NMS exists, so the timed region is the model itself. (Forward-only latencies, without post-processing, differ from end-to-end values by only ≈ 2–3 ms for YOLO and are reported alongside in the supplementary table for completeness.)

Every reported latency is a **median-of-medians**. Each measurement round runs 20 warm-up inferences and then 240 timed inferences (30 repetitions over a fixed set of 8 real COCO images), and the within-round latency is the median of the 240 samples. Because a single window is not trustworthy, the main benchmark uses one explicitly named protocol:

**Five-round interleaved latency.** All seven models are measured in **five rounds** in one campaign. Within each round, every model is measured once, and the starting model is rotated so that no detector occupies the same thermal or scheduler position in every round. The reported value is the median of the five within-round medians; the five values' minimum and maximum, together with their p95 and p99, are reported as uncertainty. The raw CSV records the round, order, start/end frequency, and timing statistics for every model.

The five-round campaign lasted about 31 min. Frequency before and after each model was logged, but the available temperature interface returned no values; this absence is recorded rather than imputed. The campaign covers only the Intel i7-14650HX platform. The YOLOv8l/YOLOv8n median latency ratio is 8.28× against an 18.9× FLOPs ratio, so the FLOPs-to-latency mismatch is reported as a seven-model empirical observation rather than a universal scaling law.

## 3.4 Accuracy reference

LAE and Pareto comparisons use a **unified full COCO val2017 evaluation**: all 5,000 images are processed with the same 640-px letterbox preprocessing, model-specific output decoding, and confidence threshold (0.001). The submission accuracy path is `scripts/evaluate_full_coco_official.py`, which writes raw detections and calls `pycocotools==2.0.11` COCOeval with the standard crowd handling and maxDets settings. The local annotation archive has been verified against the official COCO release: its archive MD5 and size match the official download, and the extracted `instances_val2017.json` is byte-for-byte identical to the working file. The older `scripts/evaluate_full_coco.py` NumPy path is retained only for development-stage relative checks. A fixed local 500-image COCO subset remains the development-stage reference for the original FP32→INT8 changes Δ; the additional complete-val INT8 confirmation is reported separately in Section 4.4.

**Which accuracy differences we are willing to assert.** Accuracy enters the Pareto analysis of Section 4.3, and a dominance edge is a claim, so we fix the rule for accepting one before looking at the outcome. The primary accuracy reference is now the unified full-val table. For close model pairs, we additionally report paired bootstrap intervals from the shared 5,000-image predictions when those prediction dumps are available; a point difference without an interval is described as an observed difference rather than a statistically separated ordering. The 500-image subset is used only for the INT8 development deltas and is not treated as an absolute reference for the full-model Pareto table.

## 3.5 LAE: a descriptive cost-weighted score

We condense the three-way accuracy–latency–size trade-off into a single scalar,

LAE(M) = mAP(M) / ( Latency_ms(M)^α × Params_M(M)^β ) ,    (1)

where α = 0.5, β = 0.3, mAP is the validated full-val COCOeval mAP50-95 reference, Latency_ms is the five-round interleaved end-to-end CPU median, and Params_M is the parameter count in millions. The exponents are fixed a priori and never fitted on the benchmarked models, since fitting them to the models they score would be circular. Their qualitative reading is standard: α = 0.5 expresses diminishing returns in latency savings below a real-time budget, and β = 0.3 expresses the sub-linear cost of parameters through memory bandwidth and footprint. Section 4 reports the resulting seven-model ranking as descriptive sensitivity analysis; it is not claimed as a validated method contribution.

Because LAE is monotone increasing in mAP and monotone decreasing in both cost axes, it cannot rank a Pareto-dominated model above its dominator. This follows from the functional form and is recorded here as a property of the score rather than as a finding of the benchmark. It takes no part in the selection protocol of Section 3.7, which is defined by the feasibility filter and the ranking within the feasible set.

## 3.6 INT8 quantization recipes

The quantized tensors follow the ONNX QuantizeLinear scale and zero-point specification [17], while available CPU kernels are provider- and operator-specific in ONNX Runtime [18].

We study post-training quantization along two independent axes, both with ONNX Runtime 1.28.0 on the same CPU and measured in the same window as the FP32 baseline.

The **quantization recipe** decides which parts of the model are quantized and how they are calibrated:

- **Dynamic (weight-only) INT8.** Weights quantized to QInt8 at runtime with FP32 activations and no calibration.
- **Naive static INT8.** Per-channel QInt8, calibrated on representative COCO val images, with the final output concatenation excluded from quantization (quantizing that node mixes box-coordinate and class-score scales and flattens class scores to zero, a toolchain trap we document). The detection head convolutions are not protected in this scheme.
- **Selective static INT8.** The naive scheme plus the entire detection head kept in FP32. We also refer to this as the head-preserving recipe.

The **export format** decides how the quantized graph is written:

- **QDQ** writes explicit QuantizeLinear and DequantizeLinear node pairs around each quantized operator.
- **QOperator** writes fused operators instead, such as QLinearConv and its relatives. It is ONNX Runtime's default format and requires unsigned 8-bit activations, so its activation type is QUInt8 where the QDQ graphs use QInt8.

The two axes are independent. The same calibrated weights can be written in either format, and the choice changes which integer kernels the runtime is able to build. Because QOperator is a runtime convention rather than a portable ONNX representation, the QDQ graphs are the ones we subject to the cross-runtime check in Section 4.4. Section 4.4 reports the two axes separately.

RT-DETR static quantization fails numerically at quantization time (zero/NaN scales), in both export formats and therefore independently of the format axis. Only its dynamic scheme is reported, and the static failure itself is reported as a toolchain-specific observation, not a claim that RT-DETR is in general unquantizable.

## 3.7 Selection look-up protocol

The measured table is consumed by an explicit, reproducible procedure (worked examples in Section 4, Table 9). Given a budget, expressed as a required mAP floor m*, a latency ceiling L* ms, and optionally a parameter ceiling P* M:

1. State the budget as bounds: mAP ≥ m*, latency ≤ L* ms, [params ≤ P* M].
2. Compute the feasible set, i.e. every benchmarked detector whose unified full-val mAP, measured end-to-end CPU latency, and parameter count all satisfy the bounds. For a hard per-frame deadline, apply the ceiling to the p99 tail figures of Section 4.1 (Table 4) rather than to the median, since the median carries no headroom for scheduler and OS jitter.
3. Apply an explicit preference rule within the feasible set: choose the highest mAP when accuracy is primary, or the lowest p99 latency when a hard deadline is primary; ties resolve toward the lower parameter count. LAE (Eq. 1, α = 0.5, β = 0.3) is reported only as a descriptive sensitivity analysis, not as a validated superior selector.
4. If the feasible set is empty, relax the tightest bound by one step the application allows (raise L*, lower m*), and repeat.

Steps 2–3 deliberately use measured latency rather than FLOPs or fitted weights. The lookup is transparent and auditable; LAE is retained only to show how a joint scalar behaves on this seven-model table.

