from transformers import DistilBertModel, DistilBertTokenizer
from pathlib import Path
import torch

# Load model and tokenizer
model_name = "distilbert-base-uncased"
model = DistilBertModel.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Export to ONNX
output_path = Path("distilbert_base_uncased.onnx")
dummy_input = tokenizer("This is a dummy input for ONNX export.", return_tensors="pt")

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    output_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=14
)
print(f"Model saved to {output_path}")
