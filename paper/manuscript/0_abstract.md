# FLOPs Latency and INT8 Export Formats in a Reproducible CPU Benchmark of YOLO and RT-DETR

**Kejie Yu**
School of Computer Science and Technology, Department of Software Engineering, Zhejiang Gongshang University, Hangzhou, China  
Corresponding author: 2162323966@qq.com  
ORCID: 0009-0002-1448-4563

---

## Abstract

FLOPs, parameters, and GPU latency can misjudge CPU detector speed. We benchmark seven CNN and Transformer detectors (YOLO11n, YOLOv8n/s/m/l, RT-DETR-l/x) with official COCO-pretrained weights on one Intel i7-14650HX CPU, 16 intra-op threads, and ONNX Runtime 1.28.0. A five-round interleaved protocol with rotated model order gives median end-to-end latencies of 33.0–410.9 ms (round range 30.4–458.7 ms); a unified full-val2017 evaluation with standard COCOeval gives mAP50-95 values from 0.367 to 0.531. FLOPs overstate CPU latency gaps: YOLOv8l/YOLOv8n is 18.9× by FLOPs but 8.28× by measured latency. RT-DETR-l uses 0.75× the parameters and 0.83× the latency of YOLOv8l, but measures 0.515 versus 0.521 mAP in the same evaluation pipeline. A fixed-exponent accuracy–latency–size score is retained only as a descriptive budget lookup because it reproduces simple single-axis rankings. INT8 behaviour depends on the tested export path and runtime: QDQ graphs are slower than FP32, whereas QOperator graphs execute more integer convolutions and reach 1.61×–1.92× FP32 speed with the naive recipe and 1.11×–1.37× with the detection head preserved. On a fixed 500-image development subset, head preservation changes mAP by −0.009 to +0.001, while the naive scheme loses 0.068–0.087; a complete-val confirmation of the head-preserving recipe changes mAP by −0.002 to −0.010 across the four YOLO scales. Static RT-DETR quantization fails numerically. Results are limited to this CPU, runtime, export paths, and 500-image INT8 uncertainty analysis, with complete-val INT8 reported as point confirmation.

**Keywords:** Object detection · CPU efficiency benchmarking · Model selection · CNN versus Transformer · INT8 quantization · Inference latency benchmarking
