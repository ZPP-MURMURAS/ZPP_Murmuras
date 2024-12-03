from transformers import AutoTokenizer, AutoModel, pipeline
from pathlib import Path
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

model_name = "google/mobilebert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

feature = "default" 
onnx_path = Path("onnx_model")
onnx_path.mkdir(parents=True, exist_ok=True)
onnx_file = onnx_path / "mobilebert.onnx"

model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)
export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=12,
    output=onnx_file,
)

print(f"Model successfully converted to ONNX at: {onnx_file}")

