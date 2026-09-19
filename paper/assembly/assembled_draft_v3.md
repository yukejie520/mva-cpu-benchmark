# FLOPs Latency and INT8 Export Formats in a Reproducible CPU Benchmark of YOLO and RT-DETR

**Kejie Yu**
School of Computer Science and Technology, Department of Software Engineering, Zhejiang Gongshang University, Hangzhou, China  
Corresponding author: 2162323966@qq.com  
ORCID: 0009-0002-1448-4563

---

## Abstract

FLOPs, parameters, and GPU latency can misjudge CPU detector speed. We benchmark seven CNN and Transformer detectors (YOLO11n, YOLOv8n/s/m/l, RT-DETR-l/x) with official COCO-pretrained weights on one Intel i7-14650HX CPU, 16 intra-op threads, and ONNX Runtime 1.28.0. A five-round interleaved protocol with rotated model order gives median end-to-end latencies of 33.0–410.9 ms (round range 30.4–458.7 ms); a unified full-val2017 evaluation with standard COCOeval gives mAP50-95 values from 0.367 to 0.531. FLOPs overstate CPU latency gaps: YOLOv8l/YOLOv8n is 18.9× by FLOPs but 8.28× by measured latency. RT-DETR-l uses 0.75× the parameters and 0.83× the latency of YOLOv8l, but measures 0.515 versus 0.521 mAP in the same evaluation pipeline. A fixed-exponent accuracy–latency–size score is retained only as a descriptive budget lookup because it reproduces simple single-axis rankings. INT8 behaviour depends on the tested export path and runtime: QDQ graphs are slower than FP32, whereas QOperator graphs execute more integer convolutions and reach 1.61×–1.92× FP32 speed with the naive recipe and 1.11×–1.37× with the detection head preserved. On a fixed 500-image development subset, head preservation changes mAP by −0.009 to +0.001, while the naive scheme loses 0.068–0.087; a complete-val confirmation of the head-preserving recipe changes mAP by −0.002 to −0.010 across the four YOLO scales. Static RT-DETR quantization fails numerically. Results are limited to this CPU, runtime, export paths, and 500-image INT8 uncertainty analysis, with complete-val INT8 reported as point confirmation.

**Keywords:** Object detection · CPU efficiency benchmarking · Model selection · CNN versus Transformer · INT8 quantization · Inference latency benchmarking

---

# 1 Introduction

Object detection increasingly runs in CPU-equipped deployments, where each frame has a latency budget against which the model must be chosen. Picking a detector is therefore an accuracy–efficiency trade-off, and the efficiency half of that trade-off is surprisingly hard to evaluate reliably. This paper measures efficiency where it can be measured carefully, on a high-core-count laptop and desktop-class x86 CPU (Intel Core i7-14650HX), and is explicit about what does and does not transfer to other platforms.

The efficiency of a detector is almost always summarised with theoretical counts (floating-point operations, or FLOPs, and parameter size) or with the GPU throughput numbers printed in its model card. All three are proxies for what actually matters on a CPU: the end-to-end time to produce detections from a real image. FLOPs count only arithmetic operations on the model's dominant layers. They ignore memory traffic, kernel and data-layout efficiency, and post-processing, so two models with identical FLOPs can differ substantially in wall-clock latency, and the gap between FLOPs and latency differs across network families (for example between convolutional heads and attention-based query decoding). GPU latencies, meanwhile, are measured on server hardware whose memory bandwidth and thread behaviour differ from a CPU, so even the ordering of detectors by speed, not just the absolute numbers, can differ between GPU and CPU.

Two further problems make published comparisons unreliable for a practitioner. First, comparisons across families are rarely controlled: CNN detectors such as YOLO and query-based detectors such as RT-DETR are typically evaluated in different codebases and runtimes, with different input pipelines, and with post-processing inconsistently included (YOLO needs non-maximum suppression, while a decoding-inclusive transformer detector does not). Second, efficiency claims about INT8 quantization are usually established on GPUs and in a specific toolchain, and there is no reason to expect them to transfer to a CPU. Our measurements below show that they do not.

Existing reports cover parts of this comparison, but they differ in hardware, runtime, post-processing, or model provenance (Section 2). We provide a controlled benchmark with the same exported official weights, ONNX Runtime backend, and explicit end-to-end accounting. Concretely, we measure seven official COCO-pretrained detectors (YOLO11n, YOLOv8-n/s/m/l of the CNN family, and RT-DETR-l/x of the Transformer family) on a single Intel i7-14650HX CPU at 16 intra-op threads with ONNX Runtime 1.28, using no fine-tuning. The benchmark is an instance study of CPU deployment, not a universal cross-platform ranking. Our contributions are:

1. **A controlled cross-family CPU benchmark under one primary interleaved protocol.** Seven official COCO-pretrained detectors (YOLO11n, YOLOv8-n/s/m/l and RT-DETR-l/x) are measured on one i7-14650HX CPU at 16 intra-op threads with ONNX Runtime 1.28, no fine-tuning, and warm-up plus median-of-medians aggregation, with end-to-end latencies from 33.0 ms (YOLOv8n) to 410.9 ms (RT-DETR-x) in the completed five-round campaign. Ranking reproducibility is assessed from the recorded rounds; no second-CPU experiment is used in the revised claims.

2. **Quantified evidence that FLOPs misreport CPU efficiency.** Theoretical FLOPs overstate measured CPU latency gaps by an amount that grows with the scale gap: the FLOPs ratio of YOLOv8l to YOLOv8n is 18.9× against an 8.28× five-round median-latency ratio. The distortion is also visible within the Transformer family. Measured latency supports a cross-family cost–accuracy comparison while the accuracy gap is made explicit: RT-DETR-l measures 0.515 versus YOLOv8l's 0.521 in the current local evaluation, while using 0.64× the FLOPs, 0.75× the parameters, and 0.83× the measured end-to-end latency.

3. **A transparent budget lookup over the measured table.** We state a constraint-first procedure that filters models by accuracy, latency, and parameter ceilings, then reports the feasible trade-off rather than hiding the choice in a fitted utility function. LAE is retained as a descriptive sensitivity analysis: on this near-monotone set it coincides with single-axis rankings, so we make no claim that it learns a better ordering.

4. **A toolchain-aware, ISA-qualified INT8 study on CPU.** Under the tested ONNX Runtime configuration, QDQ graphs are slower than FP32 because most convolutions remain on FP32 kernels surrounded by conversion nodes, whereas QOperator graphs execute more integer convolutions and are faster at two YOLO scales. Head-preserving static quantization limits the observed accuracy change on a fixed 500-image subset, while quantizing the detection head causes a much larger loss. Executed-graph inspection traces these observations to the exported graph and runtime operator support; we treat them as version- and platform-specific rather than universal properties of INT8 arithmetic.

We deliberately limit scope: one high-core-count x86 CPU class, one runtime, 640 px input, and official weights without training. Section 5.4 separates evidence about benchmarking practice from numbers that are properties of this platform. The rest of the paper is organised as follows. Section 2 reviews related work on efficiency metrics and detection benchmarks. Section 3 describes the experimental setup and latency protocols. Section 4 reports the benchmark, the descriptive budget analysis, and the INT8 study. Section 5 discusses threats to validity and usage boundaries, and Section 6 concludes.

---

# 2 Related Work

**Detector families and the efficiency numbers they report.** The two detector families we compare differ structurally in ways that matter on a CPU. Convolutional YOLO-lineage detectors [10,12] couple a CNN backbone with a lightweight head and require non-maximum suppression (NMS) as a separate post-processing step. Transformer-based detectors [2,15] replace the hand-designed head with query-based decoding: RT-DETR is NMS-free and runs a fixed, decoding-inclusive pipeline, yet its self-attention operators are memory-bandwidth-intensive and are served poorly by generic CPU kernels. Efficiency figures for both families usually originate in vendor model cards, which are not comparable across the two: YOLO cards publish FLOPs, parameters, GPU throughput, and CPU latency measured on a fixed server Xeon [12], while RT-DETR cards report GPU throughput only [15]. Juxtaposing them is an apples-to-oranges comparison, since GPU latency is measured on server hardware whose bandwidth and threading differ from a CPU, and the ordering of detectors by speed can change between the two (Sections 4 and 5 quantify this distortion).

**Benchmarking efficiency with measured latency.** Rigorous inference suites such as MLPerf [9,19] established the discipline of reporting latency under fixed hardware, batch size, precision, and repetition, across dedicated GPU and mobile suites. These harnesses cover many accelerators rather than a controlled cross-family comparison on a single CPU, and they do not address the accounting asymmetries specific to object detection, above all whether NMS and decoding are included in the reported time. Theoretical FLOPs and parameter counts remain the dominant proxies instead, because they are reproducible without hardware. Efficiency on the target device is not an arithmetic count, however. ShuffleNet V2 showed that networks with comparable FLOPs can differ markedly in measured speed, and its guidelines therefore make latency or energy on the device the metric that arbitrates efficiency [6]. This paper contributes direct evidence for that position on one CPU and under two explicitly named latency protocols (Section 3).

**Cross-family CNN-versus-transformer measurements on CPU and edge.** The closest existing measurements are two. Allmendinger et al. [1] compared YOLO and several transformer detectors for real-time weed detection, timing them on desktop CPUs and GPUs with models fine-tuned on their own agricultural dataset, an application-driven selection exercise that does not separate official pretrained weights, a single inference runtime, or the divergence between theoretical counts and measured latency. Suchý and Turčaník [11] reviewed the energy efficiency of YOLOv8l and RT-DETR-l on NVIDIA Jetson devices and, secondarily, on a Raspberry Pi, across PyTorch, TFLite, and MNN runtimes, a GPU-centred energy study limited to the two largest variants. Neither measures both families (official COCO-pretrained, spanning small and large variants alike) on a single pure-CPU rig, under one ONNX Runtime protocol, with a uniform end-to-end accounting that includes NMS for YOLO and treats RT-DETR as decoding-inclusive without NMS. To the best of our knowledge no such controlled CPU-only benchmark exists, which is the gap this paper fills, and the results that are new here are the scale dependence of the FLOPs overstatement within and across families and a toolchain-aware head-preserving INT8 comparison on the same CPU and protocol.

**Selecting a detector under explicit budgets.** Practitioners condense accuracy and cost into an ad hoc ratio such as mAP per GFLOP or mAP per millisecond. Such single-axis ratios yield a ranking but encode no trade-off among accuracy, latency, and parameters, and when the denominator is a theoretical count they inherit the distortion above. Hardware-aware selection is otherwise expressed as Pareto dominance over measured latency and accuracy. We report LAE = mAP / (Latency^0.5 × Params^0.3) as a descriptive sensitivity score, and use an explicit budget filter plus application-priority rule for deployment recommendations. Section 4 examines its ranking and margin behaviour against single-axis alternatives and reports that on this set the two agree.

**Post-training INT8 quantization on CPU.** Integer-only inference [4] is the standard route to smaller models, and ONNX Runtime exposes weight-only dynamic quantization alongside static QDQ (quantize–dequantize) schemes [16]. Two observations are well documented. First, the speed-up of INT8 is toolchain- and hardware-sensitive: on generic ONNX Runtime CPU kernels, integer models are reported to run slower than their FP32 counterparts for convolution-heavy detectors [8,7]. Second, detector heads are disproportionately sensitive to low-bit quantization, motivating head-preserving precision schemes, and transformer attention is comparatively unstable under post-training quantization. Section 4 contributes a controlled counterpart on the same CPU and protocol, and the qualitative outcomes just listed hold there. The observed QDQ/QOperator contrast is toolchain-specific; activation types were not fully matched between all artifacts, so we do not claim that format alone decides speed.

---

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

---

# 4 Results

## 4.1 Measured CPU benchmark

**Table 2** reports the primary end-to-end CPU latency benchmark. All seven detectors were measured in five interleaved rounds, with the starting position rotated across rounds, on the i7-14650HX at 16 intra-op threads. The reported statistic is the median of the five round medians; the range, p95, and p99 are computed over those five round medians. Median latency spans 33.0 ms (YOLOv8n) to 410.9 ms (RT-DETR-x), while the observed round range spans 30.4–458.7 ms. Within each family, latency increases with model scale. The LAE column is retained only as a descriptive budget lookup.

**Table 2.** Primary five-round interleaved end-to-end latency (i7-14650HX, 16 intra-op threads) and descriptive LAE (α = 0.5, β = 0.3, using the unified full-val mAP and parameter counts in Table 1).

| Model | Family | E2E median (ms) | Five-round range (ms) | p95 / p99 (ms) | Unified mAP50-95 | LAE | Rank |
|---|---|---|---|---|---|---|---|
| YOLO11n | CNN | 34.23 | 30.39–34.70 | 34.70 / 34.70 | 0.387 | 0.0493 | 1 |
| YOLOv8n | CNN | 33.00 | 30.99–40.00 | 39.78 / 39.96 | 0.367 | 0.0451 | 2 |
| YOLOv8s | CNN | 68.50 | 66.34–73.32 | 72.96 / 73.25 | 0.443 | 0.0259 | 3 |
| YOLOv8m | CNN | 158.65 | 152.06–184.88 | 181.14 / 184.13 | 0.495 | 0.0148 | 4 |
| RT-DETR-l | Transformer | 226.37 | 222.36–257.88 | 251.67 / 256.64 | 0.515 | 0.0120 | 5 |
| YOLOv8l | CNN | 273.34 | 266.84–275.63 | 275.54 / 275.61 | 0.521 | 0.0101 | 6 |
| RT-DETR-x | Transformer | 410.90 | 357.63–458.70 | 452.39 / 457.44 | 0.531 | 0.0074 | 7 |

*E2E latency = forward pass plus Python decode + NMS for YOLO, and forward pass only for RT-DETR (which is decoding-inclusive and NMS-free). The raw CSV records all five round medians and the rotated order. The temperature fields are unavailable on this Windows setup and are left empty; frequency before/after each model is recorded.*

Two pairwise observations anchor the rest of the paper. First, the two n-scale models are close in latency: YOLOv8n is faster in the five-round median (33.00 versus 34.23 ms), but the round ranges overlap and the direction changes in round 5, so we report a point-estimate ordering rather than a dominance claim. Second, RT-DETR-l is the lower-cost but lower-accuracy member of the RT-DETR-l/YOLOv8l pair in the unified full-val evaluation (0.515 versus 0.521 mAP). It uses 0.75× the parameters and 0.83× the measured latency (226.37 versus 273.34 ms). The five round-matched latency ratios are 0.848, 0.937, 0.823, 0.824, and 0.823, all below one.

**Unified full-val accuracy.** We re-evaluated all seven FP32 models on the complete 5,000-image COCO val2017 set with one preprocessing, decoding, and standard `pycocotools` COCOeval. The resulting mAP50-95 values are 0.387 (YOLO11n), 0.367 (YOLOv8n), 0.443 (YOLOv8s), 0.495 (YOLOv8m), 0.521 (YOLOv8l), 0.515 (RT-DETR-l), and 0.531 (RT-DETR-x), and are the values used in Tables 1, 2, and the Pareto analysis. The external vendor values remain useful as a sanity check but are not mixed into the main table. The full run is reproducible from `scripts/evaluate_full_coco_official.py` and `results/full_coco_map_official.csv`; the 500-image subset remains reserved for relative INT8 deltas.

Latency on this CPU is not a stable scalar. The primary evidence is therefore the rotated five-round median and its explicit range, not a single sequential window. The historical cross-day and single-window checks below are retained as supplementary context only; they do not define the main table or the LAE lookup.

**Table 3.** Cross-day reproducibility (cross-round median protocol, ranks among the three measured models).

| Model | Day 1 median (ms) | Day 1 range | Day 2 median (ms) | Day 2 range | Δ (Day 2 − Day 1) | Rank Day 1 / Day 2 |
|---|---|---|---|---|---|---|
| YOLOv8n | 31.7 | 30.6–32.6 | 33.4 | 32.0–33.5 | +5 % | 1 / 1 |
| YOLOv8m | 160.0 | 157.6–166.2 | 151.9 | 151.9–153.9 | −5 % | 2 / 2 |
| RT-DETR-l | 259.1 | 230.2–268.7 | 226.0 | 218.0–227.2 | −13 % | 3 / 3 |

*RT-DETR-l shows the largest day-to-day drift (−13 %). The ordering is what reproduces.*

**Across a second CPU.** A second-CPU measurement was available during development, but it is removed from the revised main evidence because it covered only four CNN models and a different runtime. All quantitative claims below are limited to the Intel i7-14650HX platform.

**Latency tails for hard deadlines.** Median latency is the right summary for average-throughput deployments, but a system with a hard frame deadline must budget for the tail: to meet the deadline for almost every frame, a model needs headroom above its median. To quantify that headroom we re-ran each model in one fresh contiguous window (240 timed samples per model, the same protocol and input set as Table 2) and report per-model median, p95, p99, and maximum end-to-end latency in Table 4.

**Table 4.** Latency tail statistics from a fresh single-window re-run (240 samples per model, same protocol and input images as Table 2).

| Model | Median (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|---|---|---|---|---|
| YOLO11n | 26.3 | 29.6 | 32.0 | 33.3 |
| YOLOv8n | 33.8 | 47.1 | 61.6 | 102.1 |
| YOLOv8s | 74.1 | 99.9 | 129.6 | 151.0 |
| YOLOv8m | 156.0 | 203.0 | 249.5 | 434.2 |
| RT-DETR-l | 230.4 | 252.7 | 321.1 | 478.9 |
| YOLOv8l | 276.9 | 291.4 | 302.7 | 349.7 |
| RT-DETR-x | 397.8 | 429.0 | 445.9 | 553.4 |

*Same 8 COCO images and end-to-end timing as Table 2. The medians reproduce Table 2's ordering exactly and track its values within cross-window drift (up to about ±11 %, largest on the smallest models), which is itself a reminder that absolute milliseconds are window-specific (Section 3.3).*

The tail is not uniform across models: p95 sits between 5 % (YOLOv8l) and 39 % (YOLOv8n) above the median, and p99 between 9 % (YOLOv8l) and 82 % (YOLOv8n) above it, while the maximum of every model reflects an occasional multi-frame stall. A median-based latency budget therefore does not protect a hard per-frame deadline. Following the tail-at-scale literature [3], hard real-time systems must be gated on the tail, not the average. A hard-deadline deployment should budget from the tail columns, and Section 3.7 applies the ceiling to the p99 figures when the deadline is hard.

## 4.2 FLOPs overstate measured CPU latency gaps

**Table 5** compares, for selected detector pairs, the ratio in FLOPs, parameters, and measured end-to-end CPU latency. The headline case is the small-to-large YOLO pair: the FLOPs ratio of YOLOv8l to YOLOv8n is **18.9×**, whereas the five-round interleaved latency ratio is **8.28×**, i.e. FLOPs overstate the CPU latency gap by about a factor of two. Within the Transformer family, RT-DETR-x over RT-DETR-l is 2.13× by FLOPs and 1.82× by the measured median. These are seven-model empirical pairwise trends, not a universal scaling law.

**Table 5.** Efficiency ratios for selected pairs (five-round interleaved median protocol, see Fig. 1).

| Pair (numerator / denominator) | FLOPs ratio | Params ratio | Latency ratio (forward) | Latency ratio (e2e) | Reading |
|---|---|---|---|---|---|
| YOLOv8l / YOLOv8n | 18.9 | 13.7 | – | 8.28 | FLOPs overstate e2e latency ≈ 2.3× |
| YOLOv8s / YOLOv8n | 3.27 | 3.51 | – | 2.08 | parameter ratio exceeds FLOPs ratio |
| RT-DETR-l / YOLOv8l | 0.64 | 0.75 | – | 0.83 | lower measured cost, but lower full-val mAP |

*Forward and end-to-end coincide for RT-DETR (no NMS, decoding inside the forward pass). The reported ratios use the medians in Table 2; forward-only YOLO ratios are not mixed into the primary comparison.*

We do not claim a fixed ordering among the three metrics: the parameter ratio exceeds the FLOPs ratio for the YOLOv8s/YOLOv8n pair (3.51 versus 3.27). The robust observation is directional within this seven-model set: theoretical counts overstate measured CPU latency gaps, with the largest discrepancy for YOLOv8l versus YOLOv8n. The Transformer's measured cost per counted GFLOP is also higher than the CNN's (about 2.14 versus 1.65 ms/G for RT-DETR-l against YOLOv8l), so per-FLOP efficiency figures cannot be read as CPU latency ratios.

Measured latency nonetheless preserves the cross-family conclusion that FLOPs capture only coarsely. RT-DETR-l is cheaper than YOLOv8l on every cost axis we measure: 0.64× the FLOPs, 0.75× the parameters, and 0.83× the interleaved median latency. The unified full-val mAP is lower for RT-DETR-l (0.515 versus 0.521), so this is a cost–accuracy trade-off rather than an accuracy-tied dominance claim. FLOPs distort the size of the observed cost advantage, not its direction.

## 4.3 LAE: ranking, exponent robustness, and relation to single-axis metrics

LAE scores with α = 0.5, β = 0.3 (Eq. 1) and the resulting ranks are the last two columns of Table 2. With the unified full-val accuracy reference and the interleaved medians, the descriptive ranking is YOLO11n > YOLOv8n > YOLOv8s > YOLOv8m > RT-DETR-l > YOLOv8l > RT-DETR-x, with scores from 0.0494 down to 0.0074. This is not a validated selector: the ranking is reproduced by simple single-axis orderings on this near-monotone seven-model set.

**Dominance, under a rule stated in advance (Fig. 2).** The three axes are accuracy up, latency down, parameters down. After replacing the vendor column with the unified full-val measurements, no model dominates another on all three axes: every higher-accuracy step also incurs a latency or parameter cost, and the YOLO11n/YOLOv8n pair trades accuracy against latency and size. The resulting frontier therefore contains all seven point estimates; this is a descriptive statement, not evidence that the models are statistically separated.

The closest cost pair is YOLO11n versus YOLOv8n. YOLO11n has fewer parameters and FLOPs and higher full-val mAP (0.387 versus 0.367), while the five-round latency medians are close (34.23 versus 33.00 ms) and the round direction changes once. We therefore report the point estimate without asserting dominance. RT-DETR-l versus YOLOv8l is a clear cost–accuracy trade-off in the point estimates (0.515 versus 0.521 mAP; 226.37 versus 273.34 ms).

The RT-DETR-l and YOLOv8l pair is therefore not an accuracy-tied comparison under the unified table. It remains on the three-objective frontier because its lower latency, parameter count, and FLOPs trade against the 0.014 mAP deficit.

**Rank stability over the exponents (Fig. 3(a)).** The grid analysis remains descriptive only. It shows the same ordering across the planned α ∈ [0.2, 0.8] × β ∈ [0.1, 0.5] grid when using the unified full-val accuracy column and the interleaved medians; the ranking is reproduced by simple single-axis orderings on this near-monotone set.

**Platform scope.** The revised benchmark makes no cross-platform ranking claim; the evidence is limited to the Intel i7-14650HX and ONNX Runtime 1.28.0.

**Relationship to single-axis efficiency metrics (Fig. 3(b)).** The single-axis metrics mAP/GFLOPs, mAP/ms, and mAP/params reproduce the LAE ordering on this near-monotone seven-model set, so LAE does not earn its keep by ranking differently. Its only defensible role here is to provide a transparent joint-budget lookup; it is not claimed as a superior ranking method. The margins are also distorted under theoretical counts: YOLOv8l/YOLOv8n is 18.9× by FLOPs but 8.28× by measured latency, while RT-DETR-l/YOLOv8l is 0.64× by FLOPs and 0.83× by measured latency. These figures motivate using measured latency rather than FLOPs in the selection lookup.

## 4.4 INT8 quantization on the same CPU

We report the INT8 study as a comparison against each model's FP32 baseline measured in the same run, so that the ratio and not the absolute millisecond carries the result. The four static cells come from one interleaved development run in which every variant is timed once per round with the in-round order rotated. Dynamic quantization, whose cost made interleaving impractical, comes from a separate single-window development run; its ratios are therefore quoted as within-run figures and are not pooled with the primary five-round benchmark. Two independent choices are varied. The **quantization recipe** affects accuracy and ranges over dynamic, naive static, and detection-head-preserving static. The **export path** changes the graph representation presented to ONNX Runtime, and ranges over its QDQ and QOperator paths (Section 3.6). Under the tested activation-type configurations, the two axes show different effects; this is toolchain-specific evidence rather than proof that format alone determines speed. All numbers are in Tables 6 to 8 and Fig. 4.

**Model size.** Dynamic INT8 cuts every model to roughly a quarter of its FP32 footprint (25–28 %), and naive static does the same for the four YOLO models. Selective static, which keeps the detection head in FP32, lands between the two, at 49 %, 40 %, 37 %, and 35 % of FP32 for YOLOv8 n → s → m → l.

**Table 6.** Model sizes (MB, decimal) per recipe.

| Model | FP32 | Dynamic | Naive static | Selective static | Selective / FP32 |
|---|---|---|---|---|---|
| YOLOv8n | 12.9 | 3.5 | 3.6 | 6.3 | 49 % |
| YOLOv8s | 44.9 | 11.5 | 11.7 | 18.1 | 40 % |
| YOLOv8m | 103.8 | 26.3 | 26.6 | 38.1 | 37 % |
| YOLOv8l | 175.0 | 44.2 | 44.6 | 61.5 | 35 % |
| RT-DETR-l | 131.7 | 34.2 | – (quantization fails) | – | – |
| RT-DETR-x | 265.8 | 68.1 | – (quantization fails) | – | – |

*For RT-DETR both static recipes fail numerically at quantization time (Section 3.6), so only the dynamic recipe applies to the Transformer family.*

**Accuracy.** Table 7 reports the FP32 → INT8 change in mAP50-95 on a fixed local 500-image COCO subset (same images and same code as the relative Δ of Section 3.4), arranged so that each recipe appears once under each export format. **The export format leaves accuracy where it is.** Holding the recipe fixed and changing only the format moves mAP by +0.001 to +0.005 for the naive recipe and by −0.003 to −0.001 for the head-preserving recipe, and the paired bootstrap 95 % CIs for both contrasts at both scales contain zero (B = 1000, resampling the 500 images with shared indices, Section 3.4).

**The recipe does move accuracy, and how far depends on the scale.** Naive static, which leaves the detection head in INT8, loses 0.068–0.087, a consistent ≈ 17 % across the four YOLO scales, with CIs excluding zero throughout. Head-preserving static, which keeps the whole detection head in FP32, moves mAP by −0.009 to +0.001 across the four scales. At YOLOv8s and the two larger scales the paired CIs contain zero. At YOLOv8n the change is a small but resolvable loss of 0.005 (95 % CI [−0.0125, −0.0007]). The naive minus head-preserving gap is −0.064 at YOLOv8n and −0.083 at YOLOv8s, and both paired bootstrap 95 % CIs exclude zero, which is the formal demonstration that quantizing the detection head, rather than quantization itself, is the accuracy bottleneck (Section 5.2). Dynamic quantization costs 0.006 (RT-DETR-l) to 0.016 (YOLOv8n). RT-DETR static fails numerically at quantization time in this toolchain under both export formats (Section 3.6) and is reported as such, while dynamic RT-DETR is unaffected. Adding the format axis therefore settles the question the previous version of this section could not: the two effects are controlled by different choices, so a fast quantized graph and an accurate one are not in conflict.

**Complete-val confirmation.** As a robustness check on the development-subset result, we also evaluated the four detection-head-preserving YOLO artifacts on all 5,000 val2017 images with the same official COCOeval path. Their mAP50-95 values are 0.357, 0.439, 0.492, and 0.516 for YOLOv8n, s, m, and l, respectively, versus FP32 baselines of 0.367, 0.443, 0.495, and 0.521. The corresponding changes are −0.0098, −0.0032, −0.0025, and −0.0054. Thus the direction and scale of the accuracy effect agree with the 500-image analysis, and no full-val accuracy collapse appears at any tested scale. These are complete-val point estimates; the paired bootstrap intervals reported above remain the 500-image development-stage uncertainty analysis and are not silently promoted to full-val intervals.

**Table 7.** FP32 → INT8 mAP50-95 change Δ on the 500-image subset, by recipe (columns) and export format (grouped).

| Model | FP32 (subset) | QDQ naive | QOperator naive | QDQ head-preserving | QOperator head-preserving |
|---|---|---|---|---|---|
| YOLOv8n | 0.396 | −0.069 | −0.068 | −0.005 | −0.008 |
| YOLOv8s | 0.468 | −0.085 | −0.080 | −0.002 | −0.003 |
| YOLOv8m | 0.508 | −0.087 | – | +0.001 | – |
| YOLOv8l | 0.539 | −0.087 | – | −0.009 | – |
| RT-DETR-l | 0.536 | fails | fails | fails | fails |
| RT-DETR-x | 0.539 | fails | fails | fails | fails |

*QOperator was exported for YOLOv8n and YOLOv8s (dashes mark cells not run). "fails" marks static quantization that aborts or produces a degenerate model at calibration time, under both formats. Dynamic quantization is reported in the text: −0.016 (YOLOv8n), −0.011 (YOLOv8s), −0.006 (RT-DETR-l). Dynamic INT8 of YOLOv8m/l was not run under the formal accuracy protocol because its measured slowdown (> 7×) made a full measurement window cost-prohibitive, so only model sizes are reported for those cells (Table 6), and RT-DETR-x dynamic was not run because RT-DETR-l dynamic already moved mAP by only −0.006.*

**Latency (Fig. 4, Table 8).** Whether a static INT8 model beats its FP32 baseline differs between the tested export paths and recipes; because activation types are not fully matched, this is a toolchain-specific observation rather than a format-only causal result. We measured all four recipe-format combinations in a single interleaved run, in which every variant is timed once per round with the in-round order rotated, and report the round-matched ratio of the FP32 baseline to the variant so that the clock drift common to a round cancels (Section 3.3). The FP32 baseline of YOLOv8n held within 1.2 % of its five-round median across the interleaved rounds, its lowest round being 29.435 ms against a median of 29.799 ms.

In the QDQ format no INT8 recipe is faster than FP32. Head-preserving static runs 2.1× slower than FP32 at both scales (63.4 against 29.8 ms for YOLOv8n and 137.3 against 64.4 ms for YOLOv8s), and naive static is slower still. Dynamic quantization is slower again, by 7.8× for YOLOv8n and 9.4× for YOLOv8s when measured in the single-window protocol of Section 3.3, which was used for it because its cost made the interleaved run impractical. Dynamic is accordingly absent from Fig. 4c, whose bars are all round-matched.

In the QOperator format the same two recipes are faster than FP32, and the size of that difference is the point (Fig. 4c plots all four cells of this run against parity and prints each cell's accuracy cost inside its bar). For YOLOv8n, naive static reaches 18.5 ms against 29.8 ms for FP32, a round-matched ratio of 1.61× that held in all five rounds (1.58 to 1.61×), and head-preserving static reaches 26.6 ms at 1.11× (1.10 to 1.30×). YOLOv8s reproduces the pattern at 1.92× (1.69 to 1.96×) and 1.37× (1.08 to 1.48×), and the executed-graph counts are identical across the two scales. Identical weights, identical arithmetic, and an accuracy difference the paired tests of Table 7 cannot resolve therefore differ by a factor of about four in latency, entirely through the export format. This is the cleanest causal contrast in this paper.

The 1.61× cell is a mechanism demonstration rather than a recipe to deploy. Naive static gives up 0.068 of mAP in either format (Table 7), which no accuracy budget of Section 4.5 would accept. The configuration that is both executable and accurate is QOperator export with the detection head preserved, at 1.11× to 1.37× faster than FP32 for a mAP change of −0.008 (YOLOv8n) and −0.003 (YOLOv8s). That cell is the deployable result of this section.

QOperator is not portable. OpenVINO's ONNX frontend rejects these artifacts, because ONNX Runtime emits the fused activation operators in the com.microsoft domain with dynamic element types. To separate the runtime effect from the format effect we loaded the portable QDQ models in OpenVINO 2026.3.1 into the same interleaved run as the ONNX Runtime cells, at one thread setting (16 intra-op threads for ONNX Runtime, the matching setting for OpenVINO), the same warm-up, and the same repetition count, so that every cross-runtime comparison is formed within a round. Only that one thread setting was run, and no thread sweep of OpenVINO is reported. Three round-matched comparisons come out of the run. First, OpenVINO's own FP32 baseline is the slower of the two FP32 baselines: ONNX Runtime ran it in 0.41× to 0.88× the time, a median ratio of 0.72×, below parity in all five rounds. Second, on the portable QDQ graphs OpenVINO is faster than its own FP32 baseline, with the naive cell above parity in all five rounds at 1.13× to 2.22× and the head-preserving cell above parity in four of the five, its weakest round at 0.99×. Third, on the same file OpenVINO ran the QDQ INT8 model 2.24× to 2.83× faster than ONNX Runtime did, which is the sense in which the cross-runtime advantage is at least 2.2×. Because latency under OpenVINO varies far more on this machine than under ONNX Runtime, a cell's worst round deviating from its own five-round median by 12.6 % to 60 % against 1.6 % to 11 % for the ONNX Runtime cells of the same run, we use these figures to establish a direction and a conservative bound rather than a speed-up. The direction is what the comparison was built to isolate: identical integer weights in one portable format run slower than FP32 under one runtime and faster than FP32 under another, and the runtime whose FP32 baseline is slower is the one whose INT8 wins. That places the INT8 slowdown of the QDQ column in one runtime's kernel selection rather than in the integer arithmetic or in the CPU. All latency numbers in this subsection are for ONNX Runtime 1.28.0 with its CPU execution provider on the CPU of Section 3.1, and are toolchain-specific (Section 3.6).

**Table 8.** Round-matched latency ratio (FP32 ÷ recipe, above 1 means faster) and executed-graph fusion, by recipe and export format.

| | QDQ naive | QOperator naive | QDQ head-preserving | QOperator head-preserving |
|---|---|---|---|---|
| Portable to other runtimes | yes | no | yes | no |
| YOLOv8n | 0.42× · 7/64 | **1.61× · 64/64** | 0.47× · 0/64 | **1.11× · 45/64** |
| YOLOv8s | 0.44× · 7/64 | **1.92× · 64/64** | 0.47× · 0/64 | **1.37× · 45/64** |

*Each cell gives the median round-matched ratio over five rounds, then the number of convolutions that executed as an integer QLinearConv out of 64. Every QOperator cell held above 1 and every QDQ cell below 1 in all five rounds. "Portable" means the artifact loads in an unrelated runtime. OpenVINO rejects the QOperator exports (Section 3.6), which is what limits the fast column to ONNX Runtime. Every latency ratio is measured at both scales, and every fusion count is probed at the scale it is reported for (Table S3).*

**Kernels actually executed.** A quantized ONNX graph can run integer kernels or fall back to FP32 kernels wrapped by conversion nodes, so we inspected the execution graph ONNX Runtime actually runs. We rebuilt each session under the settings used for the latency measurements (16 intra-op threads, full graph optimization), asked the runtime to dump its fully optimized model, and counted node types. Three distinct executed-graph profiles emerge, and the recipe ordering of Section 5.2 is read against them: FP32 convolutions padded by conversion nodes (static), fully integer convolutions with per-call activation quantization (dynamic), and the plain FP32 baseline. The static QDQ graphs are the first profile, with every convolution still on an FP32 kernel, except the naive variant, which fuses 7 of YOLOv8n's 64 convolutions into QLinearConv and leaves the other 57 on FP32 Conv. The dynamic graphs are the second and contain no FP32 compute kernel at all. In the QOperator format the fusion completes: at the naive recipe all 64 convolutions execute as QLinearConv with 10 conversion nodes left in the graph, at both scales probed, and the head-preserving variant fuses 45 of 64 at both scales, again with 7 conversion nodes at both (Table 8). Table S3 reports the full node census.

**Why the QDQ graph does not fuse.** The mechanism is visible both in the exported graph and in the quantizer's operator registry. PyTorch exports the SiLU activation of these detectors as a Sigmoid followed by a Mul, and in ONNX Runtime 1.28.0 neither operator carries a QDQ quantizer. The QDQ registry in the onnxruntime.quantization module lists 25 operators, among them Relu, Clip, Reshape, MaxPool and Split, and it does not list Sigmoid, Mul, Add, Concat or Softmax. The QOperator registry does list those five, which is what allows the QOperator format to fuse them. A quantized convolution therefore closes on both sides only when the operators next to it have registered quantizers, and in the naive QDQ graph the seven convolutions that do fuse are exactly those whose output feeds a Reshape, the one consumer in that chain that does have a registered quantizer.

We tested that account as a prediction rather than leaving it as an explanation. In a controlled pair of networks with identical topology, in which the only difference is the activation function, QDQ fusion moves from 4 of 5 convolutions for the ReLU version to 1 of 5 for the SiLU version, so the activation does cause the failure rather than merely accompanying it. The same rule predicts 7 of 64 fusing convolutions for YOLOv8n and 7 of 64 for YOLOv8s, matching the measured counts exactly. It is not a complete predictor of fusion rate across architectures. ResNet-18, which contains no SiLU at all, fuses only 1 of 20 convolutions because its residual additions hit the same missing-registry condition, and EfficientNet-B0 fuses 27 of 81 despite being SiLU throughout, against predictions of 9 of 20 and 2 of 81. We report the four cases in the supplementary table and state the rule as directional rather than closed-form. What it does establish is that the 7 of 64 is a property of the operator registry and the exported graph, and not an unexplained measurement.

## 4.5 Worked selection look-ups

**Table 9** applies the four-step protocol of Section 3.7 to representative budgets. Each non-empty row is read as: budget in the first column, feasible set in the second (unified full-val mAP plus measured five-round median latency plus parameters all satisfying the bounds), and the feasible detector selected by the declared application priority in the third. The final row shows the empty-set case, which triggers step 4 (relax the tightest bound and retry).

**Table 9.** Worked examples of the selection look-up protocol (latency column = Table 2, five-round interleaved median, i7-14650HX).

| Budget | Feasible set | Recommended | Why |
|---|---|---|---|
| mAP ≥ 0.37, ≤ 40 ms | YOLO11n | YOLO11n | YOLOv8n is below the full-val accuracy floor |
| mAP ≥ 0.50, no latency cap | RT-DETR-l, YOLOv8l, RT-DETR-x | RT-DETR-l | Highest descriptive LAE among the feasible models |
| params ≤ 35 M, mAP ≥ 0.45 | YOLOv8m, RT-DETR-l | YOLOv8m | Both fit the parameter cap, and the descriptive LAE prefers YOLOv8m (25.9 M, 158.65 ms) |
| mAP ≥ 0.50, ≤ 120 ms, params ≤ 30 M | ∅ | – | Infeasible, so step 4 relaxes the tightest bound, e.g. the latency ceiling to ≤ 200 ms, then retries |

*The recommendation follows the declared application priority within the feasible set; LAE is descriptive only. The four rows exercise the filter, the ranking within a multi-member feasible set, the optional parameter ceiling, and the empty-set branch of step 4.*

The table is only valid on the platform and precision reference it was built from: the latency column is the five-round i7-14650HX measurement of Table 2, and the accuracy column is the unified full COCO val2017 mAP from the local evaluation pipeline. Changing CPU, input resolution, or GPU deployment requires re-measuring the latency column before the table is used. The four-step procedure itself is unchanged. LAE sensitivity is reported separately and is not used to claim a superior selector. The role and the limits of the index are discussed in Sections 4.3 and 5.3.

---

# 5 Discussion

Results answer "what". This section asks "why", by tracing the measured gaps to their causes and by comparing our numbers with the nearest prior measurements. Two threads run through the section. First, CPU inference time is governed by memory traffic and kernel efficiency, not by the number of arithmetic operations a model performs, which is why a model with a quarter of the FLOPs can be slower (Section 5.2 is the cleanest demonstration) and why FLOPs misstate CPU gaps. Second, every efficiency comparison is an artifact of the platform, runtime, and protocol it was measured under, which is the reason the benchmark in Section 4 and the index built on it are scoped the way they are.

## 5.1 Why FLOPs misreport CPU efficiency

**Arithmetic counts, not time.** The measured cost per counted GFLOP falls as a model gets larger: with the five-round medians, YOLOv8n costs about 3.8 ms/G and YOLOv8l 1.65 ms/G, while RT-DETR-l costs about 2.14 ms/G. The Roofline model makes this precise: on a multicore CPU, achievable performance is bounded by memory traffic whenever arithmetic intensity is low [14]. On this seven-model set, measured latency ratios are smaller than FLOPs ratios within each family, and the overstatement is larger for the widest gaps. Fig. 5 reports this as an empirical pairwise trend; the 21 pairs share seven models and do not establish a universal scaling law.

**Counting conventions carry a family-dependent blind spot.** Our FLOPs count includes only Conv, ConvTranspose, MatMul, and Gemm operators (Section 3.2). Operators that consume real CPU time but are not counted, above all softmax, layer normalization, and reshape/layout bookkeeping, are more numerous in the attention blocks of RT-DETR than in the convolutional blocks of YOLO. Under the same convention, the RT-DETR-l/YOLOv8l cost–accuracy trade-off shows the consequence: RT-DETR-l costs 2.45 ms/G against 1.80 ms/G for YOLOv8l, so a cross-family FLOPs comparison inherits a blind spot that no single counting convention can remove. The vendor discrepancy for RT-DETR-l itself (105.60 G counted versus 110.2 G reported) illustrates that even within one model family, published FLOPs numbers are definition-sensitive at the few-percent level.

**Post-processing is invisible to FLOPs but real in time.** YOLO inference carries a constant Python decode plus NMS cost of roughly 2.2 to 2.5 ms per image, near 7 % of end-to-end latency on YOLOv8n and under 1 % on YOLOv8l (Section 3.3). A theoretical metric cannot see this cost at all. Because the constant is similar across YOLO scales, folding it into the end-to-end ratio already tempers the gap that FLOPs state at 18.9× (Section 4.2).

**Context against the nearest prior benchmark.** Allmendinger et al. (2025) measured RT-DETR variants on different desktop CPUs and a fine-tuned weed-detection pipeline. Our RT-DETR-l and RT-DETR-x medians are 226.4 and 410.9 ms under ONNX Runtime on the i7-14650HX. These values are not a controlled reproduction of their setup; they illustrate why CPU latency is a property of the whole runtime stack, not of the architecture name alone. We do not interpret the numerical gap as a hardware ranking.

Suchý and Turčaník (2026) review energy efficiency of YOLOv8l and RT-DETR-l across Jetson devices, Raspberry Pi, and several runtimes. As a GPU-centred energy study it does not establish a CPU-latency ordering. Our RT-DETR-l/YOLOv8l trade-off, in which RT-DETR-l is faster but lower in full-val mAP on this CPU (Section 4.1), illustrates that per-device efficiency orderings drawn on one platform do not necessarily transfer to another.

## 5.2 Why INT8 is slow here, and what makes it fast

**Quantization replaces arithmetic, not the bottleneck.** INT8 replaces the operations that FLOPs count, but on this CPU the measured bottleneck is elsewhere. In the export paths tested here, speed differed with graph representation and runtime kernel selection; because activation types were not fully matched between all QDQ and QOperator artifacts, this is an observed toolchain interaction rather than a clean causal claim that format alone decides speed. The hardware carries AVX-VNNI, so integer dot-product instructions are available to either path.

The QDQ format fails to fuse because the quantized domain never closes. A convolution becomes an integer kernel only when its output consumers are quantizable, and in ONNX Runtime 1.28.0 the QDQ registry does not list the operators that terminate a PyTorch SiLU chain, so the convolutions feeding those chains stay on FP32 kernels with quantize and dequantize nodes inserted around them. The few that do fuse are exactly those whose output feeds a registered consumer, which is what the fused counts of Section 4.4 measure. The head-preserving static recipe fuses nothing at all, since it holds the head in FP32 by construction and adds conversion nodes around everything else, which is the mechanism behind its 2.1× regression. Dynamic (weight-only) quantization avoids the registry problem entirely and runs genuine integer kernels, but it quantizes activations on the fly (DynamicQuantizeLinear), adding a full extra pass over every activation, which is why it is the slowest recipe of all.

The QOperator path has no such registry gap in this run. Exporting the same quantized weights through that path fuses the naive recipe more completely at both scales and the head-preserving recipe almost completely, while the unfused remainder corresponds to the head convolutions held back in FP32. Two cells that the paired accuracy tests cannot separate therefore end up about four times apart in latency. Because activation types are not fully matched, we attribute this observation to the tested export paths and graph configurations rather than to format alone.

These numbers are consistent with community reports that INT8 can be slower than FP32 on ONNX Runtime CPU kernels for convolution-heavy detectors (onnxruntime issues 20052 and 7), and they extend those reports rather than merely repeating them. In our tested paths, one representation produced fewer fused integer-convolution patterns than the other; because activation types were not fully matched, this is toolchain-specific evidence, not an intrinsic property of one format. The methodological point survives in sharper form: reducing arithmetic does not guarantee a CPU speed-up, and the deployment path must be measured on the target runtime.

**Portability of the two formats.** The QOperator advantage is not free. The QLinearConv family it emits is an ONNX Runtime convention rather than a portable ONNX representation, and OpenVINO 2026.3.1 refuses these artifacts outright, reporting dynamic element types on the QLinearAdd, QLinearMul, QLinearSigmoid and QLinearSoftmax nodes. The OpenVINO run of Section 4.4 makes the same point from the other side: identical integer weights in the same portable QDQ format run faster than FP32 under a second runtime, while that runtime's own FP32 baseline is the slower of the two FP32 baselines in every round. We report its ratios as a conservative lower bound only, since over that run the OpenVINO cells deviated from their five-round medians by 12.6 % to 60 %, against 1.6 % to 11 % for the ONNX Runtime cells. The practical reading is that QDQ is the portable choice and QOperator the fast one on this runtime, and that a practitioner who needs both should expect to export twice.

**Why the detection head decides accuracy.** The two static recipes differ only in where quantization is applied, and the paired tests of Section 4.4 place the gap between them outside bootstrap noise. The loss of full-static INT8 is therefore attributable to the detection head rather than to quantization of the backbone and neck: the head is where box coordinates and class scores are predicted from small tensors, and quantizing it corrupts those predictions. The same mechanism produces the toolchain trap documented in Section 3.6, where quantizing the final output concatenation flattens class scores to zero. RT-DETR static quantization fails earlier still, at quantization time and in both export formats, with zero or NaN scales that trace to the backbone-encoder region rather than to the detection output. We report this as a toolchain observation. Dynamic RT-DETR, which avoids QDQ entirely, is the one recipe whose accuracy cost on this set is negligible.

## 5.3 LAE as a decision tool, and its boundaries

**The exponent choice encodes an engineering position.** Latency on an edge CPU is a hard budget: a detector that misses the per-frame deadline is unusable regardless of its accuracy, whereas accuracy beyond the application need has no value either. The α = 0.5 exponent expresses diminishing returns in latency savings below a real-time budget, and β = 0.3 expresses the sub-linear cost of parameters through memory bandwidth and footprint.

The exponents are fixed before any measurement and are never fitted on the seven models they rank, which would be circular. Section 4.3 shows the ranking is insensitive to them over a broad grid, so the recommended model does not hinge on the choice.

**What LAE is and is not.** LAE is a descriptive score within this benchmark, not a validated method contribution or universal ordering. Its ranking agrees with simple ratios on the seven-model set, so we do not claim a practical advantage. The recommended deployment rule is transparent: filter by hard accuracy/latency/parameter budgets, then select according to the application's stated primary objective (accuracy or tail latency); LAE is an optional sensitivity report.

**A property of the score that this benchmark does not exercise.** Because LAE is monotone increasing in mAP and decreasing in both cost axes, it cannot rank a Pareto-dominated model above its dominator (Section 3.5). On this set the monotonicity property is not exercised, since no benchmarked model is Pareto-dominated under the rule of Section 3.4. We record it as a property of the functional form and do not rely on it anywhere in the selection procedure.

The degenerate corner of the exponent grid (α and β near zero), where the index collapses toward an accuracy-only ranking, is a property to state rather than to fix: with no cost weight, the score is not supposed to encode cost. In practice it will be most useful to a developer with a hard latency ceiling who wants to know which feasible detector to take, which is the look-up protocol of Section 4.5.

**Boundaries of use.** Absolute LAE scores are comparable only within one platform and one accuracy reference: the latency column is the five-round i7-14650HX measurement and the accuracy column is the unified full COCO val2017 evaluation (Section 3.4). No portability claim is made across platforms. Changing runtime, input resolution, weight provenance, or dataset requires recomputing the inputs to the index, not re-arguing its form.

## 5.4 Threats to validity, and how we bounded them

**Measurement windows.** Laptop-CPU latency drifts with turbo state, scheduler, and thermal headroom. We bound this with 20 warm-up inferences, 240 timed samples per round, five interleaved rounds, rotated order, and explicit round ranges plus p95/p99 of the round medians. The primary benchmark does not claim that one contiguous sequential run estimates a stationary latency distribution; it reports the observed campaign and its uncertainty. Frequency was logged per model, while temperature was unavailable through the local interface.

**Software and toolchain.** All numbers are for ONNX Runtime 1.28.0 on Windows with 16 intra-op threads. The INT8 results in particular are toolchain-specific and are reported as such, and the RT-DETR static failure should not be read as a claim that RT-DETR is in general unquantizable. The one cross-runtime comparison in the paper, the OpenVINO measurements of Section 4.4, is the weakest evidence we report: at the single thread setting it was run under, its cells deviated from their five-round medians by 12.6 % to 60 %, against 1.6 % to 11 % for the ONNX Runtime cells of the same run, so we use it only to establish the direction of the runtime effect and a conservative lower bound, and we do not treat any of its ratios as a speed-up. The instability itself is a finding about measuring INT8 latency across runtimes, and it is one more reason the single-runtime protocol of Section 3.3 is the right frame for the rest of this paper. Re-running on another runtime or OS will change absolute latencies and could change small absolute differences between adjacent models. The structural conclusions (FLOPs overstate CPU latency gaps and explicit budgets yield an auditable lookup) depend on direction and rank. The INT8 contrast remains toolchain-specific. What would test the INT8 conclusion hardest is a runtime built around integer dot-product kernels on a different ISA, which we did not run: OpenVINO covers the runtime axis on this machine, but the ARM `sdot` and `udot` path and the TFLite and MNN engines remain untested, and the same export-format effect should be re-established there rather than assumed. The hardware axis is likewise narrower than it looks: because this CPU's ISA is the one described in Section 3.2, a CPU without integer dot-product instructions is not covered by these measurements.

**YOLO post-processing.** The timed region for YOLO includes a Python decode and NMS pipeline, a deployment-representative but implementation-specific choice (Section 3.3). A vendor-optimized C++ NMS would shave roughly 2 to 3 ms per image from the YOLO end-to-end numbers, which would slightly change absolute gaps and ratios between YOLO and RT-DETR. It would not change the FLOPs-versus-latency argument, which is computed between YOLO models on the same YOLO accounting, nor the within-YOLO dominance claims. Forward-only latencies are reported alongside for exactly this reason.

**Accuracy reference and scope.** The revised manuscript uses one full-val code path for the FP32 accuracy and LAE/Pareto table, while the local 500-image subset remains the source of INT8 paired uncertainty intervals (Section 3.4). The four-model complete-val selective INT8 run is now included as a point-estimate confirmation, not as a substitute for full-val bootstrap intervals or a complete survey of all quantization recipes. The full-val result does not remove the need for a second hardware platform, but it does remove the vendor/subset mismatch from the main table. All claims are scoped to official COCO-pretrained weights at 640 px without fine-tuning, one runtime, and one CPU class, and are evidence about how to benchmark and select a detector on a CPU, not universal throughput statements.

**Which conclusions transfer.** We separate conclusions that are evidence about how to benchmark on a CPU from numbers that are properties of this one platform. The directional FLOPs-versus-latency observation and the graph-level INT8 mechanism are hypotheses supported by this seven-model, one-runtime experiment; they should be retested on other CPUs and runtimes. No cross-platform ranking claim is made. Absolute latencies, the magnitude of cross-model gaps, and LAE scores are specific to this x86 CPU at 16 intra-op threads. When the target platform differs, the latency and accuracy columns should be re-measured rather than the method re-argued.

---

# 6 Conclusion

This paper asked whether the numbers routinely used to compare object detectors, theoretical FLOPs and vendor latency, tell a practitioner how fast a model will actually run on a CPU. The measured answer, on one CPU under two named protocols, is that FLOPs do not, and that the discrepancy is systematic rather than incidental.

The conclusions that follow are an abstraction of the measured table, not a restatement of it. In five rotated interleaved rounds on one Intel i7-14650HX, the YOLOv8l/YOLOv8n FLOPs ratio is 18.9× while the measured median-latency ratio is 8.28×; RT-DETR-x/RT-DETR-l is 2.13× by FLOPs and 1.82× by latency. These are empirical seven-model trends, not a universal scaling law, and they illustrate why CPU inference time cannot be inferred from arithmetic counts alone. RT-DETR-l is faster than YOLOv8l in every matched round and uses fewer parameters, but its unified full-val mAP is lower (0.515 versus 0.521), so the pair is a cost–accuracy trade-off rather than a free improvement. LAE is retained as a descriptive joint-cost sensitivity analysis; deployment recommendations use explicit budget filtering and application priorities rather than claiming a superior index.

**Implications.** For efficiency reporting, a model card listing FLOPs and parameters without measured CPU latency under a named warm-up protocol cannot support cross-family or cross-platform claims. Cross-round medians test stability across independently sampled windows, while a single contiguous window places every model in one pass under comparable conditions. For model selection, report the feasible set under explicit accuracy, latency-tail, and parameter budgets, then apply the deployment priority. For a practitioner considering INT8 on a CPU, quantization is a size lever and can be an accuracy lever when the detection head is preserved, but it is not an automatic speed lever. In these experiments, QDQ and QOperator paths showed different outcomes under ONNX Runtime; because activation types were not fully matched, the result is toolchain-specific evidence, not proof that format alone determines speed.

**Limitations and scope.** Section 5.4 sets out the scope within which the conclusions hold. In brief, they are evidence about how to benchmark a detector on a CPU and are bounded by one runtime on one CPU class, by a seven-model set with one instance per scale, and by INT8 paired uncertainty intervals measured on a 500-image development subset. The complete-val selective INT8 run strengthens the point estimate but does not turn the INT8 study into a full-val bootstrap analysis. The mechanistic account in Section 5 is interpretation built on the measured quantities, not a separately measured quantity.

**Future work.** The next release step is to extend the complete-val INT8 confirmation to the naive and dynamic control recipes and to add paired full-val bootstrap intervals for the closest model pairs. Re-measuring the same seven-model table on further CPUs and runtimes, with the Transformer models included, would test whether the scale-dependent FLOPs overstatement is a property of this CPU class or of CPU execution in general. On the quantization side, profiling dequantization and node-insertion overhead on a runtime with mature integer kernels would separate a toolchain deficiency from a genuine ceiling of integer inference for detectors.

---

# Statements and Declarations

**Funding.** The author declares that no funds, grants, or other support were received during the preparation of this manuscript.

**Competing interests.** The author has no relevant financial or non-financial interests to disclose.

**Author contributions.** The author confirms sole responsibility for the study conception and design, data collection, analysis and interpretation of results, and manuscript preparation.

**Data availability.** The reproducibility package is available at https://github.com/yukejie520/mva-cpu-benchmark in release `v0.2.1` and at https://doi.org/10.5281/zenodo.22839651. It includes the benchmark scripts, manuscript source, figures, environment specification, and curated `release_data/` package with the five-round latency logs, standard full-val FP32 mAP results, complete-val selective INT8 confirmation, thread-sensitivity measurements, relative-path metadata, and model hashes. COCO images and third-party model weights are not redistributed; the release includes download instructions, image list, and SHA-256 hashes. Paired bootstrap intervals for close full-val model pairs remain follow-up work. The 500-image INT8 paired intervals remain explicitly development-stage, while the four-model complete-val selective INT8 results are reported as point estimates in Section 4.4.

**Use of AI tools.** An AI assistant was used for language and editorial assistance and to cross-check that the numbers quoted in the prose match the archived measurement files. The study design, the measurements, the analysis, and the conclusions are the author's, and the author takes full responsibility for the content of the publication.

---

# Supplementary Material

**FLOPs Latency and INT8 Export Formats in a Reproducible CPU Benchmark of YOLO and RT-DETR**  
Machine Vision and Applications  
Kejie Yu — School of Computer Science and Technology, Department of Software Engineering, Zhejiang Gongshang University, Hangzhou, China  
Corresponding author: 2162323966@qq.com  
ORCID: 0009-0002-1448-4563

*Not part of the numbered sections. Referenced from Sections 3.3 and 4.4.*

## Table S1. Fusion rate of the QDQ export format, predicted versus measured

The mechanism advanced in Section 4.4 states that a quantized convolution fuses in the QDQ format only when the operators consuming its output carry a registered QDQ quantizer. We tested that statement as a prediction on six graphs. Prediction is the count of convolutions whose consumers are all present in the ONNX Runtime 1.28.0 QDQ registry, excluding graph outputs. Total is the count of convolutions in the FP32 graph of the same model, and Fused is the count that execute as QLinearConv after export. All six graphs are exported by the same PyTorch path and quantized with the same calibration set of 64 images.

| Graph | Activation | Total conv | Predicted fused | Measured fused | Rate | Outcome |
|---|---|---|---|---|---|---|
| Tiny control, ReLU | ReLU | 5 | 5 | 4 | 4/5 | agrees to within one |
| Tiny control, SiLU | SiLU | 5 | 1 | 1 | 1/5 | exact |
| YOLOv8n | SiLU | 64 | 7 | 7 | 7/64 | exact |
| YOLOv8s | SiLU | 64 | 7 | 7 | 7/64 | exact |
| ResNet-18 | ReLU | 20 | 9 | 1 | 1/20 | overpredicts by 8 |
| EfficientNet-B0 | SiLU | 81 | 2 | 27 | 27/81 | underpredicts by 25 |

The two tiny control graphs are identical in topology and differ only in the activation function. Moving that single factor takes fusion from 4 of 5 to 1 of 5, which establishes the activation as a cause of the failure rather than a correlate. The rule reproduces the measured 7 of 64 exactly on both YOLO detectors, which are the models this paper reports.

It is not a closed-form predictor across architectures, and the last two rows are the evidence for that. ResNet-18 contains no SiLU at all yet fuses only 1 of its 20 convolutions: its residual additions are themselves absent from the QDQ registry, so the prediction of 9 counts the Add blockers as harmless when they are not. EfficientNet-B0 is SiLU throughout and fuses 27 of 81 against a prediction of 2, so the registry account overstates the damage a SiLU chain does in that graph. We therefore report the rule as directional. It explains why the YOLO QDQ graphs are almost entirely unfused, and it predicts the two controlled cases exactly, but the fusion rate of an arbitrary architecture requires reading the exported graph rather than applying the rule.

## Table S2. Forward-only and end-to-end latency

Same single-window sequential measurement as Table 2 of the main text, reported here at the two accounting boundaries so that the cost of the YOLO post-processing path can be read directly. Forward-only is the ONNX Runtime `run` call alone. End-to-end adds the Python decode and NMS pipeline for the YOLO family.

| Model | Forward median (ms) | Forward range (ms) | End-to-end median (ms) | End-to-end range (ms) | End-to-end − forward (ms) |
|---|---|---|---|---|---|
| YOLO11n | 27.4 | 27.3–27.9 | 29.6 | 29.5–30.2 | 2.1 |
| YOLOv8n | 29.3 | 27.0–29.4 | 31.5 | 29.3–31.6 | 2.1 |
| YOLOv8s | 65.3 | 65.0–66.8 | 67.4 | 67.3–69.1 | 2.1 |
| YOLOv8m | 149.3 | 149.1–152.4 | 151.8 | 151.0–154.6 | 2.5 |
| RT-DETR-l | 227.7 | 224.7–229.7 | 227.7 | 224.7–229.7 | 0.0 |
| YOLOv8l | 270.6 | 261.6–271.3 | 272.5 | 263.8–273.5 | 2.0 |
| RT-DETR-x | 395.5 | 395.0–397.6 | 395.5 | 395.0–397.6 | 0.0 |

*RT-DETR is decoding-inclusive and NMS-free, so its two columns coincide by construction and the zero difference is not a measurement of a cheap post-processing path. On the YOLO family the difference is a near-constant 2.0 to 2.5 ms across a thirteen-fold span of forward latency, which is the post-processing floor that Section 5.1 uses to explain why the end-to-end ratio is smaller than the forward-only ratio at the widest scale gap.*

## Table S3. Executed-graph node census

The node counts behind the three executed-graph profiles of Section 4.4. Each session was rebuilt under the latency-measurement settings (16 intra-op threads, full graph optimization), the runtime was asked to dump its fully optimized model, and node types were counted. "Conv as integer" is the number of convolutions that execute as QLinearConv or ConvInteger, "Conv on FP32" the number left on an FP32 Conv kernel, and "Conversion" the number of quantize/dequantize nodes inserted into the graph.

| Graph | Recipe | Conv as integer | Conv on FP32 | Conversion | Total nodes |
|---|---|---|---|---|---|
| YOLOv8n | FP32 | 0 | 64 | 0 | 219 |
| YOLOv8n | dynamic | 64 | 0 | 59 | 493 |
| RT-DETR-l | dynamic | 110 | 0 | 117 | 1416 |
| YOLOv8n | QDQ naive | 7 | 57 | 636 | 884 |
| YOLOv8n | QDQ head-preserving | 0 | 64 | 511 | 747 |
| YOLOv8s | QDQ naive | 7 | 57 | 636 | 884 |
| YOLOv8s | QDQ head-preserving | 0 | 64 | 511 | 747 |
| YOLOv8l | QDQ head-preserving | 0 | 104 | 927 | 1295 |
| YOLOv8n | QOperator naive | 64 | 0 | 10 | 251 |
| YOLOv8n | QOperator head-preserving | 45 | 19 | 7 | 245 |
| YOLOv8s | QOperator naive | 64 | 0 | 10 | 251 |
| YOLOv8s | QOperator head-preserving | 45 | 19 | 7 | 245 |

*Every row of this census is a direct probe of the graph named in it. Both QOperator cells are identical at both scales, the naive graph fusing all 64 convolutions with 10 conversion nodes and the head-preserving graph fusing 45 with 7, and the QDQ head-preserving graph is likewise identical at both scales (0 fused, 511 conversion nodes). The QDQ naive graph is the one probe whose convolution topology is shared rather than coincidentally equal: it fuses 7 of 64 at both scales (Table S1).*

---

# Figures

![Fig. 1](results/figures/Fig1.png)

**Fig. 1.** Cross-family efficiency mismatch under the five-round interleaved protocol (Section 3.3). Each panel plots, for the four detector pairs YOLOv8l/YOLOv8n, YOLOv8s/YOLOv8n, RT-DETR-x/RT-DETR-l, and RT-DETR-l/YOLOv8l, the ratio of the listed metric of the numerator model over the denominator model on a log scale, with a parity reference at 1 and bars coloured by the family of the numerator (blue for YOLO, orange for RT-DETR). (a) Theoretical FLOPs. (b) Parameters. (c) Measured end-to-end latency on the i7-14650HX at 16 intra-op threads, with whiskers propagated from the five-round min–max ranges. FLOPs overstate the measured latency gap of the widest YOLO pair (18.9× versus 8.28×), while the cross-family pair RT-DETR-l/YOLOv8l lies below parity on all three metrics.

![Fig. 2](results/figures/Fig2.png)

**Fig. 2.** Accuracy–latency–size analysis using the five-round interleaved medians (Section 3.3), i7-14650HX at 16 intra-op threads. Horizontal axis: measured end-to-end CPU latency on a log scale. Vertical axis: unified standard COCOeval mAP50-95 on all 5,000 val2017 images. Marker area is proportional to parameter count. Each point is labelled with its model name, so the family is recoverable without colour. The star marks the descriptive LAE top-1 (YOLO11n at α = 0.5, β = 0.3); Pareto statements remain descriptive point-estimate comparisons.

![Fig. 3](results/figures/Fig3.png)

**Fig. 3.** Descriptive LAE robustness and margin behaviour under the fixed exponents of Eq. 1. (a) Exponent-sensitivity heat map over the wide grid α ∈ [0.05, 1.0], β ∈ [0.05, 0.6], using the unified standard COCOeval accuracy column and the five-round latency medians. (b) Efficiency margins computed per counted GFLOP (purple) and per measured millisecond (blue) for the pairs YOLOv8n over YOLOv8l (13.3 versus 8.28) and YOLOv8l over RT-DETR-l (0.64 versus 0.83). The figure is a sensitivity analysis, not evidence of a superior selection method.

![Fig. 4](results/figures/Fig4.png)

**Fig. 4.** Post-training INT8 quantization of the seven detectors under ONNX Runtime 1.28.0 with its CPU execution provider (i7-14650HX, 16 intra-op threads), each recipe measured in the same run as its FP32 baseline, so that the ratios are formed within a single measurement window (Section 4.4). (a) Model size of each recipe as a fraction of the FP32 model per detector and recipe. The dynamic and naive-static recipes fall to roughly a quarter of FP32, while selective static, which keeps the detection head in FP32, remains at 35–49 %. (b) Change in mAP50-95 on the fixed 500-image COCO subset from the FP32 baseline to each recipe: dynamic changes it by −0.016 to −0.006, selective static by −0.009 to +0.001, and naive static loses 0.069–0.087 in the QDQ format, which is the format this panel plots, against FP32 subset baselines of 0.396–0.539. Across both export formats the naive recipe loses 0.068–0.087, the wider range reflecting the QOperator cells that only panel (c) carries. RT-DETR static recipes fail at quantization time in this toolchain and are marked with a cross. (c) Round-matched latency ratio for the four export-format and recipe cells that were run to completion, where the bar height is each round's FP32 median divided by that same round's variant median, above 1 means faster than FP32, and the whisker is the spread over the five interleaved rounds. The number printed inside each bar is that cell's change in mAP50-95, which for the QDQ cells repeats panel (b) and for the QOperator cells is the corresponding value of Table 7. The same weights, the same arithmetic, and an accuracy cost that the paired tests of Table 7 cannot separate therefore land on opposite sides of parity depending only on the export format. Only YOLOv8n and YOLOv8s appear in panel (c): the QOperator format was not exported for YOLOv8m or YOLOv8l, and no usable QOperator export exists for RT-DETR, so these two bars per cell do not extend to the other five detectors. Dynamic INT8 is absent from panel (c) because its cost made an interleaved run impractical. Its ratios are same-window figures reported in the text.

![Fig. 5](results/figures/Fig5.png)

**Fig. 5.** FLOPs-to-latency empirical pairwise trend across all 21 unordered pairs of the seven-model set, fitted by ordinary least squares on logs. Each pair is oriented so that the numerator has the larger FLOPs count, the horizontal axis is the FLOPs ratio and the vertical axis the measured end-to-end CPU latency ratio, both on a log scale, and the dashed line is parity. Latencies are the five-round interleaved medians on the i7-14650HX at 16 intra-op threads. The solid line is the descriptive fit (γ ≈ 0.70, cluster-bootstrap 95 % interval [0.59, 0.85], R² ≈ 0.942, n = 21). Because γ < 1, the measured latency ratio grows sublinearly with the FLOPs ratio in this seven-model sample. The 21 pairs share models and are therefore not independent; this is a descriptive trend on one platform, not a universal scaling law.

The figure files are supplied separately as vector PDFs and 300-dpi PNG previews; captions are provided only in the manuscript text.

---

# References

[1] Allmendinger, A., Saltık, A. O., Peteinatos, G. G., Stein, A., & Gerhards, R. (2025). Assessing the capability of YOLO- and transformer-based object detectors for real-time weed detection. *Precision Agriculture*, 26(3), Article 52. https://doi.org/10.1007/s11119-025-10246-0

[2] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers. In *Proceedings of the 16th European Conference on Computer Vision (ECCV 2020)*. Lecture Notes in Computer Science, vol. 12346, pp. 213–229. https://doi.org/10.1007/978-3-030-58452-8_13

[3] Dean, J., & Barroso, L. A. (2013). The tail at scale. *Communications of the ACM*, 56(2), 74–80. https://doi.org/10.1145/2408776.2408794

[4] Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018)*, pp. 2704–2713. https://doi.org/10.1109/CVPR.2018.00286

[5] Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. In *Proceedings of the 13th European Conference on Computer Vision (ECCV 2014)*. Lecture Notes in Computer Science, vol. 8693, pp. 740–755. https://doi.org/10.1007/978-3-319-10602-1_48

[6] Ma, N., Zhang, X., Zheng, H.-T., & Sun, J. (2018). ShuffleNet V2: Practical guidelines for efficient CNN architecture design. In *Proceedings of the 15th European Conference on Computer Vision (ECCV 2018)*. Lecture Notes in Computer Science, vol. 11218, pp. 122–138. https://doi.org/10.1007/978-3-030-01264-9_8

[7] Microsoft. (2023). *onnxruntime*, GitHub issue #16009: QUInt8 vs a basic ONNX. Retrieved September 8, 2026, from https://github.com/microsoft/onnxruntime/issues/16009

[8] Microsoft. (2024). *onnxruntime*, GitHub issue #20052: INT8 quantized model run slower than FP32 model. Retrieved September 8, 2026, from https://github.com/microsoft/onnxruntime/issues/20052

[9] Reddi, V. J., Cheng, C., Kanter, D., Mattson, P., Schmuelling, G., Wu, C.-J., et al. (2019). MLPerf inference benchmark. arXiv:1911.02549. https://doi.org/10.48550/arXiv.1911.02549

[10] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)*, pp. 779–788. https://doi.org/10.1109/CVPR.2016.91

[11] Suchý, I., & Turčaník, M. (2026). Review of large YOLOv8 and RT-DETR energy efficiency on edge devices for real-time detection. *Scientific Reports*, 16(1), Article 10908. https://doi.org/10.1038/s41598-026-46453-6

[12] Ultralytics. (2023). *Ultralytics YOLOv8* [Computer software]. https://github.com/ultralytics/ultralytics

[13] Ultralytics. (2024). *Ultralytics YOLO11* [Computer software]. https://github.com/ultralytics/ultralytics

[14] Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4), 65–76. https://doi.org/10.1145/1498765.1498785

[15] Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., & Chen, J. (2024). DETRs beat YOLOs on real-time object detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)*, pp. 16965–16974. https://doi.org/10.1109/CVPR52733.2024.01605

[16] ONNX Runtime. (2026). Quantize ONNX models. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

[17] ONNX. (2026). QuantizeLinear operator specification. https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html

[18] ONNX Runtime. (2026). Operator kernels. https://onnxruntime.ai/docs/reference/operators/OperatorKernels.html

[19] MLCommons. (2026). MLPerf Inference submission guide. https://docs.mlcommons.org/inference/submission/
