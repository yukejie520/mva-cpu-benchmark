# FLOPs Latency and INT8 Export Formats in a Reproducible CPU Benchmark of YOLO and RT-DETR

**Kejie Yu**
School of Computer Science and Technology, Department of Software Engineering, Zhejiang Gongshang University, Hangzhou, China  
Corresponding author: 2162323966@qq.com  
ORCID: 0009-0002-1448-4563

---

## Abstract

Choosing a detector by FLOPs, parameters, or GPU latency can misjudge its CPU speed. We benchmark seven CNN and Transformer detectors (YOLO11n, YOLOv8n/s/m/l, RT-DETR-l/x) using official COCO-pretrained weights without fine-tuning, on one Intel i7-14650HX CPU with 16 intra-op threads and ONNX Runtime 1.28.0. Under warm-up and median-of-medians protocols, end-to-end latency spans 29.6–395.5 ms at official mAP50-95 of 0.373–0.548. FLOPs overstate measured CPU latency gaps: for YOLOv8l/YOLOv8n the FLOPs ratio is 18.9× against 9.4× under the cross-round protocol and 8.7× in one sequential window. RT-DETR-l uses 0.75× the parameters and 0.84×–0.87× the latency of YOLOv8l, while their published accuracies are statistically indistinguishable under our local check. A fixed-exponent accuracy–latency–size score is reported as a descriptive budget lookup; on this near-monotone set it produces the same ordering as single-axis metrics and is not claimed as a superior ranking method. For INT8, the observed result depends on the export path and runtime. QDQ graphs are slower than FP32 in the tested ONNX Runtime configuration because most convolutions remain on FP32 kernels surrounded by conversion nodes. QOperator graphs execute more integer convolutions and reach 1.61×–1.92× FP32 speed for the naive recipe and 1.11×–1.37× with the detection head preserved. On a fixed 500-image subset, head preservation changes mAP by −0.009 to +0.001, whereas the naive scheme loses 0.068–0.087. Static RT-DETR quantization fails numerically in this toolchain. These findings are platform- and version-specific; they motivate re-measurement on the target CPU rather than universal speed claims.

**Keywords:** Object detection · CPU efficiency benchmarking · Model selection · CNN versus Transformer · INT8 quantization · Inference latency benchmarking
