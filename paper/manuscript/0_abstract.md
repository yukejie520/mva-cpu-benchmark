# FLOPs Latency and INT8 Export Formats in a Reproducible CPU Benchmark of YOLO and RT-DETR

**Kejie Yu**
School of Computer Science and Technology, Department of Software Engineering, Zhejiang Gongshang University, Hangzhou, China  
Corresponding author: 2162323966@qq.com  
ORCID: 0009-0002-1448-4563

---

## Abstract

Choosing a detector by FLOPs, parameters, or GPU latency can misjudge its CPU speed. We benchmark seven CNN and Transformer detectors (YOLO11n, YOLOv8n/s/m/l, RT-DETR-l/x) using official COCO-pretrained weights without fine-tuning, on one Intel i7-14650HX CPU with 16 intra-op threads and ONNX Runtime 1.28.0. In a five-round interleaved protocol with rotated model order, median end-to-end latency spans 33.0–410.9 ms; the across-round range is 30.4–458.7 ms. Official mAP50-95 values span 0.373–0.548. FLOPs overstate measured CPU latency gaps: the YOLOv8l/YOLOv8n FLOPs ratio is 18.9×, versus an 8.28× ratio in the interleaved medians. RT-DETR-l is 0.75× the size and 0.83× the latency of YOLOv8l, while their published accuracies differ by only 0.001. A fixed-exponent accuracy–latency–size score is retained as a descriptive budget lookup; on this seven-model set it reproduces the single-axis ordering and is not claimed as a superior selector. For INT8, the observed result depends on the export path and runtime: QDQ graphs are slower than FP32 in the tested ONNX Runtime configuration, whereas QOperator graphs execute more integer convolutions and reach 1.61×–1.92× FP32 speed for the naive recipe and 1.11×–1.37× with the detection head preserved. On a fixed 500-image development subset, head preservation changes mAP by −0.009 to +0.001, whereas the naive scheme loses 0.068–0.087. Static RT-DETR quantization fails numerically in this toolchain. These findings are limited to the stated CPU, runtime, export paths, and development-stage accuracy subset; they motivate re-measurement on the target deployment platform.

**Keywords:** Object detection · CPU efficiency benchmarking · Model selection · CNN versus Transformer · INT8 quantization · Inference latency benchmarking
