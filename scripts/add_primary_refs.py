from pathlib import Path
root=Path(__file__).resolve().parents[1]
f=root/"paper/manuscript/2_related_work.md"; s=f.read_text(encoding="utf-8")
s=s.replace("schemes. Two observations", "schemes [16]. Two observations")
f.write_text(s,encoding="utf-8")
f=root/"paper/manuscript/3_methods.md"; s=f.read_text(encoding="utf-8")
needle="## 3.6 INT8 quantization recipes"
s=s.replace(needle,needle+"\n\nThe quantized tensors follow the ONNX QuantizeLinear scale and zero-point specification [17], while available CPU kernels are provider- and operator-specific in ONNX Runtime [18].")
f.write_text(s,encoding="utf-8")
f=root/"paper/manuscript/references.md"; s=f.read_text(encoding="utf-8")
if "[16] ONNX Runtime" not in s:
 s += "\n[16] ONNX Runtime. (2026). Quantize ONNX models. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html\n\n[17] ONNX. (2026). QuantizeLinear operator specification. https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html\n\n[18] ONNX Runtime. (2026). Operator kernels. https://onnxruntime.ai/docs/reference/operators/OperatorKernels.html\n\n[19] MLCommons. (2026). MLPerf Inference submission guide. https://docs.mlcommons.org/inference/submission/\n"
f.write_text(s,encoding="utf-8")
