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

