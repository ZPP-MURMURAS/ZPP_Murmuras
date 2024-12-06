# Comparison of AI Platforms

# Hugging Face 
## Useful links
- Pricing: https://huggingface.co/pricing 
- This link has a tutorial on how to start training a model on DGX Cloud and it looks to be intuitive and simple to use. There is also an auto train option: https://huggingface.co/blog/train-dgx-cloud 
- https://huggingface.co/docs/google-cloud/en/index 
- https://huggingface.co/docs/hub/en/advanced-compute-options 

## Some Information
Limits: (1) Here is a link to all the limits and prices for the HF spaces. Notably: 4x Nvidia A10G - large with 96GB GPU memory, 184 GB CPU memory for $10.80 per hour. Hugging Face Spaces allows one to host ML demo apps. 
https://huggingface.co/docs/hub/spaces-overview 

Pricing: There is a list of CPUs and GPUs and their hourly rates in the first link (Pricing). Examples of GPUs include: NVIDIA T4, NVIDIA L40S and 	8x Nvidia L40S. GPU memory ranges from 14GB to 384GB and varies by price of course. 
https://huggingface.co/pricing 

Supported libs: 
Here you can find all the libraries supported by HF. (1, 2).  Some of the notable ones include Transformers, TensorFlowTTS, and BERTopic. 
It also appears that you can upload your own libraries to the HF hub (3). This provides the following benefits: free model hosting, built-in file versioning - even for huge files - made possible by Git-LFS, and Usage metrics for all models ran with your library. 
https://huggingface.co/docs/hub/en/models-libraries
https://huggingface.co/docs/hub/en/datasets-libraries
https://huggingface.co/docs/hub/en/models-adding-libraries

Flops: I found a package called ”️Model (Transformers) FLOPs and Parameter Calculator” which was developed in Huggingface Space specifically for mainly transformers model to compute FLOPs. This could be useful for calculating flops. However, I could not find any other information on this matter. Here is a link to the package: https://huggingface.co/spaces/MrYXJ/calculate-model-flops. 

Cost-efficiency (flop per dollar?): Not found

What they offer: 
HF Spaces: https://huggingface.co/docs/hub/spaces-overview 
Hugging Face Hub: https://huggingface.co/docs/hub/en/index -> offers many models and datasets that we can utilize.

Notes: There is an autotrain option on the site so that you can select any model on the site (including BERT) and click autotrain. This could be useful if we are set on working with BERT or any model from HF. However this is a no-code feature so we lack control. 
There is also AutoTrain Advanced which can also be run locally through its intuitive UI or accessed via the Python API provided in the autotrain-advanced package. Link (1)
https://huggingface.co/docs/autotrain/en/index 

Conclusion: This is good if we want to use a pre-trained model from HF such as BERT, since it supports models from HF and libraries used for them. It is versatile as it allows us to upload our own libraries and train our own models, however 


# Replicate
## Some links
- https://replicate.com/ 
- https://replicate.com/pricing 
- https://replicate.com/docs/get-started/google-colab 
- Tutorial for fine-tuning a model: https://replicate.com/docs/get-started/fine-tune-with-flux 
- Changing the hardware the model runs on: https://replicate.com/docs/topics/models/hardware 

GPUs: Full list here (1). Some notable examples are: 
Nvidia A40 (Large) GPU, 48GB of GPU RAM, 72GB RAM
Nvidia L40S GPU, 48GB of GPU RAM, 65GB RAM
8x Nvidia H100 GPU, $0.012200/sec: “Flux fine-tunes run on H100s”
https://replicate.com/pricing 

Supported Libraries: here is a list of all supported libraries (1). There are official libraries for inter alia python and CLI interface and there are several community-supported libraries.
https://replicate.com/docs/reference/client-libraries  

Notes: They allow for running code from python (1) and google colab (2). You can use local files and URLs as inputs and seems to be aligned with our needs as we can deploy our code quite easily. 
https://replicate.com/docs/get-started/python
https://replicate.com/docs/get-started/google-colab 
The docs are really good in my opinion.

Conclusion: I think this is also fine as they offer really powerful GPUs and allow for training your own model. From what I understand, you import replicate as a library in python then train your model. This could be good as one has a lot of control over the process. 

# Modal Labs
## Useful links
- https://modal.com/pricing 
- Fine tuning: https://modal.com/use-cases/fine-tuning 

GPUs: Nvidia H100, Nvidia A100, 80 GB, Nvidia T4

Pricing: we pay for CPU time and memory used.

What they offer: Real-time metrics and logs (useful for evaluating our models). Integration with popular tools which allow one to monitor experiment results with Weights and Biases and visualize training progress using TensorBoard.

Supported Libraries: seems to allow using any libraries one can import (like in Google Colab). Flexible framework integration: we can use ML fine-tuning frameworks, like Hugging Face, PyTorch, and Axolotl…or write our own training loop. (1)
https://modal.com/use-cases/fine-tuning 

Notes: We can scale up to hundreds of multi-GPU fine-tuning runs in just a few seconds with a single function call and we can share environments with each other without having to set up the environments locally. This could be good for our collaboration.

Conclusion: Seems to have the same functionality as above and has a nice UI. It allows for placing datasets and code and launching it. It is nice that they offer tools to monitor and visualize our models and “experiments” which could be useful but maybe not exactly necessary for our case. 

# Jarvis Labs
## Useful Links
- https://jarvislabs.ai/ 
- https://jarvislabs.ai/pricing 
- https://docs.jarvislabs.ai/environment/ 

GPUs: (all are Nvidia GPUs) H100 SXM, A100, RTX6000Ada, A6000, A5000, RTX5000

Notes:  Using Jarvislabs, you can create one of the following instances: PyTorch, FastAI, Tensorflow. This seems quite restrictive and may not suit our needs. (1)
https://docs.jarvislabs.ai/environment/ 
We can use VS Code, Jupyter Labs or SSH to start coding into our instance. This seems to also give use a lot of flexibility and does not restrict us to no-code solutions. 

Conclusion: It is good however not exactly special compared to other sites. The main difference will be the types of GPUs offered and the feature that we can adjust the number of GPUs or storage capacity. 

# LangSmith
## Useful links
- https://www.langchain.com/langsmith 
- https://docs.smith.langchain.com/self_hosting 
- https://docs.smith.langchain.com/observability/how_to_guides 
- Docs: https://docs.smith.langchain.com/ 

Conclusion: This website is really difficult to navigate and I could not find any information on the GPUs they offer or anything of value regarding this task. I don’t think this is worth our time. LangSmith is mainly focused on LLM development which is also not exactly aligned with our goals.This seems to trace calls to a model to monitor the way it is working and whether there are any errors or bottlenecks. This could be good in later stages of our project.


# Predibase
## Useful links
- https://predibase.com/ 
- List of models: https://predibase.com/models  
- https://predibase.com/pricing 
- https://predibase.com/fine-tuning 
- https://docs.predibase.com/user-guide/inference/byom 
- https://predibase.com/blog/fine-tuning-zephyr-7b-to-analyze-customer-support-call-logs 
- Evaluation: https://docs.predibase.com/user-guide/fine-tuning/evaluation 

GPUs: Nvidia A100, 

Models supported: We can deploy any open-source model on Predibase and then fine-tune it, which would be good if we only want to fine-tune a BERT (or another) model. 

What they offer: A way to evaluate our solutions although it does not seem to be as advanced as Modal. Fine-tuning uses A100s by default but you can choose other hardware to further optimize for cost or speed.

Conclusion: Again, it will probably work fine if we choose it however it doesn’t have any exceptional features compared to other sites. I found it difficult to find any tutorials on how to use this site but that is not a major issue. It is fine, but not exceptional.

# OpenPipe
## Useful links
- https://openpipe.ai/ 
- https://openpipe.ai/pricing 
- https://docs.openpipe.ai/introduction 

Notes: Really intuitive UI; there is a place to drop data sets, logs, evaluations and where you can fine-tune your model. 
According to their website you can only choose which model to fine-tune from their curated set (1). This may be too restrictive for our purposes. 
https://docs.openpipe.ai/features/fine-tuning/quick-start 

GPUs: I could not find information

What they offer: Ways to evaluate the models which is a useful tool. 

Conclusion: This does not seem to be what we are looking for as I cannot find any tutorial or any information on uploading our own model. This would limit us to the models that they have chosen. 

# API credits from OpenAI
## Useful links
- https://openai.com/api/pricing/ 
- https://platform.openai.com/docs/models 

Conclusion: Seems to just be about training GPTs and other models they offer. I can’t find any way to upload my own model which is worrying. If we want to train a GPT model then this would be the way to go in my opinion as they provide updates to the models. However, this platform does not seem to be versatile and does not allow for easy uploading of our own models etc.

# BrainTrust 
- https://www.braintrust.dev/ 
- https://www.braintrust.dev/pricing 
- https://www.braintrust.dev/docs/guides/self-hosting 
- This is kind of similar to our task: https://www.braintrust.dev/docs/cookbook/recipes/ClassifyingNewsArticles#classifying-news-articles   

What they offer: evaluations of our models. It does not seem to be as advanced as Modal’s but it is also a good feature. Worryingly, they say “Many of the advanced capabilities of Braintrust involve defining and calling custom code functions. Currently, Braintrust supports defining functions in JavaScript/TypeScript and Python, which you can use as custom scorers or callable tools.” which does not seem to imply a high level of control over our model and code. (1) They also offer real-time monitoring (2). 
https://www.braintrust.dev/docs/reference/functions 
https://www.braintrust.dev/docs/start#real-time-monitoring 

GPUs: nothing mentioned

Notes: There is a prompt playground which could be useful if we need to do prompt engineering at some point.

Conclusion: What is unique to this site is their prompt engineering playground which could be useful if we want to make ChatGPT generate our data sets. This does not seem to fit our needs 

# Google Colab
## Useful links
https://stackoverflow.com/questions/47109539/what-are-the-available-libraries-within-google-colaboratory 
Notes: I will keep this brief since all of us have experience with Colab. It has a simple and intuitive UI, similar to Google Docs. It allows us to run code on the cloud and it is easy to upload and manage data sets. It requires no set-up. From what I gathered, you can download pretty much any library you want which is a pro. However, it does not seem like you can choose the type of GPU a model is trained on, unlike some other sites from this list. 

GPUs: NVIDIA Tesla K80 with 12GB of VRAM

Conclusions: most of these sites offer similar services with the difference being the quality of their documentation, UI, GPUs offered and the monitoring tools they offer. Google Colab seems to be the best option as we are already accustomed to it however Modal, Jarvislabs and Replicate are also pretty good. 

