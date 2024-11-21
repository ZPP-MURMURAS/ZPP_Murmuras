# Model
In the `model` directory there is a `convert_distilbert_qa.py` script to convert the pre-trained BERT model to TorchScript format and output its tokenizer vocabulary.

# App
In the `mobile_app` directory there is an example of a question-answering mobile app using the model created by the script. You need to copy `model/qa360_quantized.ptl` file and `model/qa360_quantized_tokenizer/vocab.txt` file to the `mobile_app/app/src/main/assets` directory to run the application.

# References
The app and model creation script are taken from <https://github.com/pytorch/android-demo-app/tree/master/QuestionAnswering>. I modified the model creation script to create the `vocab.txt` file and updated dependencies in the app. To learn more about the app, please refer to the original repository. To learn more about exporting a huggingface model to TorchScript, see <https://huggingface.co/docs/transformers/torchscript>.
