# ONNX Runtime Text Classifier for Android

This repository contains an Android application implementing a text classification model using ONNX Runtime. The app processes user-input text, tokenizes it, and runs inference using a pretrained BERT-based model to produce classification results. While the code executes successfully, it may not yield the intended results due to potential model, preprocessing, or inference pipeline issues. 

In this project I used the [distilbert](https://huggingface.co/distilbert/distilbert-base-uncased) model which yielded the best results although they weren't amazing. This is a [screenshot](./screenshots/i_am_example.jpg) from the app. The app completes the words missing in the way that can be seen (here)[https://huggingface.co/distilbert/distilbert-base-uncased]. It does work quite long and when running it, you might have to adjust the amount of RAM you allocate (I changed my settings from 4GB to 6GB - search 'RAM' into your Android phone settings and then click 'RAM Plus' and adjust if necessary).

Please keep in mind that the following scripts are examples and you can modify them to suit your specific needs. 

# Converting the model
```
pip install transformers onnx onnxruntime onnxruntime-tools
```
Then run the scripts in the `python_scripts` directory in alphabetical order (A-export-model-to-onnx.py, B-...). Move the .onnx file into the app/src/main/res/raw directory. Change the file names in the code if necessary. Now the project should come with .onnx files so you don't have to change anything.

Download any missing libraries etc. 


