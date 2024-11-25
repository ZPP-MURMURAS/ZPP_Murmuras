# ONNX Runtime Text Classifier for Android

This repository contains an Android application implementing a text classification model using ONNX Runtime. The app processes user-input text, tokenizes it, and runs inference using a pretrained BERT-based model to produce classification results. While the code executes successfully, it may not yield the intended results due to potential model, preprocessing, or inference pipeline issues. 

In this project I used the [mobile bert](https://huggingface.co/google/mobilebert-uncased) and [distilbert](https://huggingface.co/distilbert/distilbert-base-uncased) models which yielded the best results although they weren't amazing. This is a [screenshot](screenshot1.jpg) from the app. The app completes the words missing in the way that can be seen (here)[https://huggingface.co/distilbert/distilbert-base-uncased]. 

# Installing a model
```
git lfs install  # Install Git Large File Storage if you haven't already.
git clone https://huggingface.co/google/mobilebert-uncased
pip install huggingface_hub
```

Run the following script in Python.
```
from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/mobilebert-uncased")
```

# Converting the model
```
pip install transformers onnx onnxruntime onnxconverter-common
```
Then run this script.
```
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
```

Download any missing libraries etc. 


