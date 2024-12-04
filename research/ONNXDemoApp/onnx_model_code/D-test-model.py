'''
from transformers import DistilBertTokenizer
import onnxruntime as ort

# Load the tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Load quantized ONNX model
quantized_model_path = "distilbert_base_uncased_quantized.onnx"
session = ort.InferenceSession(quantized_model_path)

# Prepare inputs
dummy_input = tokenizer("This is a test input.", return_tensors="np")
inputs = {
    "input_ids": dummy_input["input_ids"].numpy(),
    "attention_mask": dummy_input["attention_mask"].numpy()
}

# Run inference
outputs = session.run(None, inputs)
print(outputs)

'''
from transformers import DistilBertTokenizer
import onnxruntime as ort

# Load the tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Load quantized ONNX model
quantized_model_path = "distilbert_base_uncased_quantized.onnx"
session = ort.InferenceSession(quantized_model_path)

# Prepare inputs
dummy_input = tokenizer("This is a test input.", return_tensors="np")
inputs = {
    "input_ids": dummy_input["input_ids"],  # Already a NumPy array
    "attention_mask": dummy_input["attention_mask"]  # Already a NumPy array
}

# Run inference
outputs = session.run(None, inputs)
print(outputs)

