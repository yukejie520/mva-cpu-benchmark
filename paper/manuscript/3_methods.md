# 3 Methods

## 3.1 Detectors and weights

We benchmark seven object detectors spanning the two architectural families, all official COCO-pretrained weights used without fine-tuning:

- **CNN family (YOLO-lineage).** YOLO11n [13] and YOLOv8 n / s / m / l [12].
- **Transformer family.** RT-DETR-l / x (HGNetv2 backbone) [15].

All models come from the same source repository (Ultralytics) and were exported to ONNX with the vendor's export pipeline, so that inference runtime, input preprocessing, and weight provenance are identical across families. The set spans roughly one order of magnitude in latency (smallest YOLO11n at 29.6 ms measured end-to-end on our CPU and largest RT-DETR-x at 395.5 ms), 2.7–66.4 M parameters, and 6.5–225.0 GFLOPs (Table 1). YOLO11n, released after YOLOv8, is included because its position on the accuracy–latency frontier is unknown a priori and our index is validated on the final seven-model set.

**Table 1.** Benchmark detectors (official Ultralytics COCO-pretrained weights, 640 px, no fine-tuning).

| Model | Family | Params (M) | FLOPs (G) | Official mAP50-95 |
|---|---|---|---|---|
| YOLO11n | CNN | 2.66 | 6.54 | 0.395 |
| YOLOv8n | CNN | 3.19 | 8.74 | 0.373 |
| YOLOv8s | CNN | 11.20 | 28.60 | 0.449 |
| YOLOv8m | CNN | 25.93 | 78.94 | 0.502 |
| RT-DETR-l | Transformer | 32.83 | 105.60 | 0.530 |
| YOLOv8l | CNN | 43.71 | 165.15 | 0.529 |
| RT-DETR-x | Transformer | 66.37 | 224.98 | 0.548 |

*FLOPs counted at runtime on the exported ONNX graph (2 × MACs over Conv/ConvTranspose/MatMul/Gemm, so RT-DETR-l is definitionally lower than the vendor's 110.2 G, see §3.2). Official mAP = COCO val2017, as published by each model's vendor.*

## 3.2 Hardware and software configuration

All latency measurements were conducted on one platform: a laptop-class Intel Core i7-14650HX CPU (16 physical cores / 24 threads, 16 GB RAM) running Windows 11, with **turbo boost enabled**, matching how a laptop CPU is deployed in practice. No GPU was used at any point. The "no accelerator" constraint is deliberate, since our claims concern pure-CPU deployment. Inference ran under ONNX Runtime 1.28.0 (CPU execution provider) at **16 intra-op threads** and batch size 1, with 640 px letterboxed COCO input. We chose 16 rather than all 24 intra-op threads after a sweep: throughput improved from 8 to ~16 intra-op threads, while opening all 24 increased end-to-end latency by about 28 % (hyper-thread contention). The CPU supports AVX2 and AVX-VNNI, the 256-bit integer dot-product extension, and does not support AVX-512 (CPUID flags avx2 and avx_vnni, with OpenVINO 2026.3.1 reporting INT8 among this device's optimisation capabilities). The ISA therefore does offer integer dot-product instructions, which matters for reading Section 4.4: an INT8 slowdown on this machine cannot be attributed to their absence. Because turbo frequencies drift with load and temperature, core frequency was logged at 1 Hz with the Windows Performance Counter (`Processor Information\Processor Frequency`) during the measurement windows. In the longest window (≈ 19 min, the single-window run of Section 3.3) the logged frequency averaged 2.07 GHz with a within-window standard deviation of 0.12 GHz, and 35 of 251 samples (13.9 %) fell below 1.9 GHz and 47 (18.7 %) below 2.0 GHz, concentrated in one sustained dip of ≈ 183 s rather than spread uniformly. The log carries no model markers, so a dip cannot be attributed to an individual detector. We therefore treat rankings rather than absolute milliseconds as the primary evidence wherever within-window drift is non-trivial.

**FLOPs.** Since FLOPs is a central object of this study, we do not take vendor card values but count FLOPs at runtime on the exported ONNX graphs: only Conv / ConvTranspose / MatMul / Gemm operators are counted as 2 × MACs (aligning with the Ultralytics convention), with tensor shapes read from one forward pass. Skip connection operators and normalization/attention-bookkeeping ops are excluded. Counts reproduce vendor values within ≈ 0.5 % for YOLO models. The one discrepancy, RT-DETR-l (105.6 GFLOPs counted vs. 110.2 G reported), is definitional and is reported in Table 1 with the counted value.

## 3.3 Latency measurement protocols

Latency on a laptop CPU is not a stable scalar: absolute timings drift by tens of percent across windows as the scheduler, memory state, and thermal headroom change. Any single round of measurement is therefore a sample of a distribution, not the quantity itself. We report **end-to-end latency** with the deployment-accounting asymmetry made explicit: for YOLO detectors the timed region includes the forward pass plus Python post-processing (decode + non-maximum suppression), because a deployed YOLO cannot emit detections without it. For RT-DETR the forward pass already contains query decoding, and no NMS exists, so the timed region is the model itself. (Forward-only latencies, without post-processing, differ from end-to-end values by only ≈ 2–3 ms for YOLO and are reported alongside in the supplementary table for completeness.)

Every reported latency is a **median-of-medians**. Each measurement round runs 20 warm-up inferences and then 240 timed inferences (30 repetitions over a fixed set of 8 real COCO images), and the within-round latency is the median of the 240 samples. Because a single window is not trustworthy, we define two explicitly named protocols, used for different purposes throughout the paper:

**Cross-round median latency.** Each model is measured in **three independent rounds**, each a separate window, and the reported value is the median of the three within-round medians, with the cross-round min–max range reported as uncertainty. Separate rounds smooth window-to-window turbo/scheduler noise and give the most honest absolute numbers. This protocol feeds the headline efficiency-mismatch ratios (Fig. 1). It covers six of the seven detectors: YOLO11n was added to the model set after the cross-round campaign and is reported under the single-window protocol only.

**Single-window sequential latency.** All seven models are measured back-to-back **within one continuous window** (≈ 19 min, gap-free), each still as three rounds of median-of-medians. All models therefore run under one contiguous load history, so cross-model differences are not diluted by window-to-window drift, and LAE scores, Pareto comparisons, and the selection look-up all use this protocol (Figs. 2 and 3). Within-window frequency drift is logged and reported (Section 3.2), and the heaviest model, RT-DETR-x, is the most sensitive to it, so its absolute value carries the widest frequency caveat.

The main-platform model order was not recorded, and model-level frequency intervals cannot be reconstructed post hoc. The Pi 5 protocol uses a rotated model order. Consequently, run-order control is not comparable between the two platforms.

The two protocols can disagree in ratio: the YOLOv8l/YOLOv8n measured ratio is 9.4× under cross-round and 8.7× under single-window measurement. This is a designed scope difference rather than measurement error: cross-round medians average three windows sampled independently across the day, while the single-window protocol holds all seven models in one contiguous pass, so the two estimate different objects by construction. What is robust is the ranking and the direction of the effect: against the 18.9× FLOPs ratio of the same pair, both protocols find a latency ratio near 9×, an overstatement of about a factor of two for this widest pair under either protocol. Section 4 shows the overstatement is scale-dependent, from about 1.3× for the closest pair to about 2.0× for the widest. In general we take within-platform rankings as the robust object of study, and absolute cross-protocol numbers as range-bounded.

## 3.4 Accuracy reference

For LAE and all Pareto comparisons we use each model's **official mAP50-95 on COCO val2017** [5] as published by its vendor: YOLOv8 values from the Ultralytics model table, RT-DETR from [15], and YOLO11n from the Ultralytics YOLO11 model table. These accuracy values, all measured at 640 px on the full 5,000-image set, are the numbers reviewers recognize. We additionally measured mAP50-95 on a fixed local 500-image COCO subset, but only to compute the FP32→INT8 change Δ: a same-images, same-code relative difference is immune to subset bias. The same measurements serve one further purpose, a consistency check on the vendor column of Table 1 (Section 4.1), which does not place them in any absolute role. The 500-image results are development-stage evidence and are not presented as replacements for a complete re-evaluation. Subset values are never used as an absolute accuracy or in LAE (see §4).

**Which accuracy differences we are willing to assert.** Accuracy enters the Pareto analysis of Section 4.3, and a dominance edge is a claim, so we fix the rule for accepting one before looking at the outcome. An edge is asserted only when both of two conditions hold. First, the sign of the accuracy difference must agree across the two references we have, the vendor-published value and our own 500-image same-pipeline measurement. A pair whose two references point in opposite directions is directionally undetermined and carries no edge. Second, the paired bootstrap 95 % interval of that difference, formed by resampling the 500 images under shared indices, must have a lower bound at or above zero, so that the positive direction is established and not merely unrefuted. Of the two ways to write the second condition we take the stronger one deliberately: retaining an edge whenever the opposite direction has not been demonstrated would accept the null hypothesis. The rule is stated here because it is applied uniformly to all 21 unordered model pairs in Section 4.3, where we report how many edges survive it rather than only the pairs that decide the figure.

## 3.5 LAE: a descriptive cost-weighted score

We condense the three-way accuracy–latency–size trade-off into a single scalar,

LAE(M) = mAP(M) / ( Latency_ms(M)^α × Params_M(M)^β ) ,    (1)

where α = 0.5, β = 0.3, mAP is the official COCO mAP50-95, Latency_ms is the single-window measured end-to-end CPU latency, and Params_M is the parameter count in millions. The exponents are fixed a priori and never fitted on the benchmarked models, since fitting them to the models they score would be circular. Their qualitative reading is standard: α = 0.5 expresses diminishing returns in latency savings below a real-time budget, and β = 0.3 expresses the sub-linear cost of parameters through memory bandwidth and footprint. Section 4 shows the resulting seven-model ranking is insensitive to the exponent choice over a broad grid.

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
2. Compute the feasible set, i.e. every benchmarked detector whose official mAP, measured end-to-end CPU latency, and parameter count all satisfy the bounds. For a hard per-frame deadline, apply the ceiling to the p99 tail figures of Section 4.1 (Table 4) rather than to the median, since the median carries no headroom for scheduler and OS jitter.
3. Apply an explicit preference rule within the feasible set: choose the highest mAP when accuracy is primary, or the lowest p99 latency when a hard deadline is primary; ties resolve toward the lower parameter count. LAE (Eq. 1, α = 0.5, β = 0.3) is reported only as a descriptive sensitivity analysis, not as a validated superior selector.
4. If the feasible set is empty, relax the tightest bound by one step the application allows (raise L*, lower m*), and repeat.

Steps 2–3 deliberately use measured latency rather than FLOPs or fitted weights. The lookup is transparent and auditable; LAE is retained only to show how a joint scalar behaves on this seven-model table.

