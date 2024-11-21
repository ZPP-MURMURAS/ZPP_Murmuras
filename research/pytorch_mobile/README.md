# Model
In the `model` directory, there is a `convert_distilbert_qa.py` script to convert a pre-trained BERT model to TorchScript format and save it along with its tokenizer vocabulary.

# App
In the `mobile_app` directory, there is an example of a question-answering mobile app using the model created by the script. You need to copy `model/qa360_quantized.ptl` file and `model/qa360_quantized_tokenizer/vocab.txt` file to the `mobile_app/app/src/main/assets` directory to run the application.

# References and Notes
The app and model creation script are taken from <https://github.com/pytorch/android-demo-app/tree/master/QuestionAnswering>. I modified the model creation script to create the `vocab.txt` file and updated dependencies in the app. To learn more about the app, please refer to the original repository. To learn more about exporting a Hugging Face model to TorchScript, see <https://huggingface.co/docs/transformers/torchscript>. It should be noted that the Hugging Face documentation mentions exporting only Transformers models and that, as of 21.11.2024, the preface of the "Export to TorchScript" section states:
> This is the very beginning of our experiments with TorchScript and we are still exploring its capabilities with variable-input-size models. It is a focus of interest to us and we will deepen our analysis in upcoming releases, with more code examples, a more flexible implementation, and benchmarks comparing Python-based codes with compiled TorchScript.

I would also advise to pay attention to the use of the `trace` method instead of `script` (<https://pytorch.org/docs/stable/jit.html>).
