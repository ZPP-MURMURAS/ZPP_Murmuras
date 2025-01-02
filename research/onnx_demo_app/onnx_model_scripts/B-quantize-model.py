from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize the ONNX model
quantized_model_path = "distilbert_base_uncased_quantized.onnx"
quantize_dynamic(
    model_input="distilbert_base_uncased.onnx",
    model_output=quantized_model_path,
    weight_type=QuantType.QUInt8
)
print(f"Quantized model saved to {quantized_model_path}")
