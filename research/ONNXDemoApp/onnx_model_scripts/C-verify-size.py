import os
quantized_model_path = "distilbert_base_uncased_quantized.onnx"
size_in_mb = os.path.getsize(quantized_model_path) / (1024 * 1024)
print(f"Quantized model size: {size_in_mb:.2f} MB")
