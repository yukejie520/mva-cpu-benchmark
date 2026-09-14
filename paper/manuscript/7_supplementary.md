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

*Every row of this census is a direct probe of the graph named in it. Both QOperator cells are identical at both scales, the naive graph fusing all 64 convolutions with 10 conversion nodes and the head-preserving graph fusing 45 with 7, and the QDQ head-preserving graph is likewise identical at both scales (0 fused, 511 conversion nodes). The QDQ naive graph is the one probe whose convolution topology is shared rather than coincidentally equal: it fuses 7 of 64 at both scales (Table S1). The 59 conversion nodes of the dynamic YOLOv8n graph are per-call activation quantizers (DynamicQuantizeLinear), the extra pass over activations that Section 5.2 attributes its cost to.*
