# Comparison of AI Platforms

# Hugging Face 
- https://huggingface.co/docs/hub/en/billing
-> Cloud integration with AWS, Azure, and Google Cloud.
-> Compute services with Spaces, Inference Endpoints and the Serverless Inference API.
- https://huggingface.co/blog/train-dgx-cloud 
-> You can find Train on DGX Cloud on the model page of supported Generative AI models. It currently supports the following model architectures: Llama, Falcon, Mistral, Mixtral, T5, Gemma, Stable Diffusion, and Stable Diffusion XL. 
-> This link has a tutorial on how to start training a model on DGX Cloud and it looks to be intuitive and simple to use. There is also an autotrain option. 
- https://huggingface.co/docs/google-cloud/en/index
-> Hugging Face built Deep Learning Containers (DLCs) for Google Cloud customers to run any of their machine learning workload in an optimized environment, with no configuration or maintenance on their part. These are Docker images pre-installed with deep learning frameworks and libraries such as 🤗 Transformers, 🤗 Datasets, and 🤗 Tokenizers. The DLCs allow you to directly serve and train any models, skipping the complicated process of building and optimizing your serving and training environments from scratch.
- https://huggingface.co/docs/hub/en/advanced-compute-options
Conclusion: Pricing is for the usage rather than a fixed-price per time period which could be a pro or con depending on our usage. HF offers powerful NVIDIA GPUs which we could use. There is integration with PyTorch-based models and HF libraries which is also a pro. 

# Replicate 
- https://replicate.com/ 
- https://replicate.com/pricing 
-> They offer a larger variety of GPUs compared to HF. 
-> Pay for use (like HF). 
- https://replicate.com/docs/get-started/google-colab 
Conclusion: seems to be better integrated than HF. 

# Modal Labs
- https://modal.com/pricing
- The starter plan only allows up to three users. Upgrading to the organization plan will unlock additional seats.
- To be honest it seems to be as good as Replicate. 

# Jarvis Labs
- https://jarvislabs.ai/
- https://jarvislabs.ai/pricing 
- https://docs.jarvislabs.ai/environment/ 
-> Using Jarvislabs, you can create one of the following instances: PyTorch, FastAI, Tensorflow
-> Not as flexible as HF

# LangSmith
- https://www.langchain.com/langsmith 
- https://docs.smith.langchain.com/self_hosting 
Side note: hard to find info on this website which isn’t a good sign 
- https://docs.smith.langchain.com/observability/how_to_guides 
Conclusion: not really worth our time in my opinion, really difficult to find information on the site and does not mention any GPUs that they offer. 

# Predibase
- https://predibase.com/ 
- https://predibase.com/models: l
-> Solar Mini, Solar Pro Preview
-> Llama 3.2 (1B, 3B; Instruct and non-Instruct)
-> Llama 3.1 8B (Instruct and non-Instruct)
-> Mistral-7b, Mistral-7b-instruct-v0.1 and v0.2
-> Mistral Nemo 12B 2407 (Instruct and non-Instruct)
-> Mixtral-8x7B-Instruct-v0.1
-> Codellama 13B Instruct, Codellama 70B Instruct
-> Zephyr 7B Beta
-> Gemma 2 (9B, 27B; Instruct and non-Instruct)
-> Phi 3.5 Mini Instruct
-> Phi 3 4k Instruct
-> Qwen 2.5 (1.5B, 7B, 14B, 32B; Instruct and non-Instruct)
-> Qwen 2 (1.5B; Instruct and non-Instruct)
-> Any OSS Model from Huggingface (best effort)
- https://predibase.com/pricing 
- https://predibase.com/fine-tuning 
-> Fine-tuning uses A100s by default but you can choose other hardware to further optimize for cost or speed.
- https://predibase.com/blog/fine-tuning-zephyr-7b-to-analyze-customer-support-call-logs 
- https://docs.predibase.com/user-guide/inference/byom -> can upload any model from HF or other repos 
- https://docs.predibase.com/user-guide/fine-tuning/evaluation -> provides some way to evaluate your model. 
-> This is quite good since we can easily evaluate our models and I could not find this in any other site. 
Conclusion: pretty good and versatile. It is possible to launch any model and fine-tune it. They provide some way of evaluating our models.

# OpenPipe
- https://openpipe.ai/ 
- https://openpipe.ai/pricing -> pricing per tokens (?)
Once you log into the site, you can see an intuitive UI which will make our process of fine tuning and working with models easier. 
See `in open_pipe_ui.png`
- https://docs.openpipe.ai/introduction 
Conclusion: I think this is worth our time since it seems to be easy to work with and it is versatile. 

# API credits from OpenAI
- https://openai.com/api/pricing/ 
- https://platform.openai.com/docs/models 
Seems to just be about training GPTs and other models they offer. I can’t find any way to upload my own model which is worrying. If we want to train a GPT model then this would be the way to go in my opinion as they provide updates to the models. However, this platform does not seem to be versatile and does not allow for easy uploading of our own models etc. 

# BrainTrust
- https://www.braintrust.dev/ 
- https://www.braintrust.dev/pricing 
- https://www.braintrust.dev/docs/guides/self-hosting 
-> there is a prompt playground which could be useful if we need to do prompt engineering at some point 
- https://www.braintrust.dev/docs/cookbook/recipes/ClassifyingNewsArticles#classifying-news-articles -> kind of similar to our task; shows it could be used for our project
I think that this is good however not better than HF or Replicate. 

In my opinion the most worth our time (in descending order) are: OpenPipe, HF, Replicate, Predibase, 
