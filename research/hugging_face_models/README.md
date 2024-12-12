# How to use hugging face tutorials
This folder contains two notebooks; the first one showcases how we can download a sample dataset and model, how to fine-tune it
and how to upload it to the Hugging Face model hub. The second notebooks presents a simple way of evaluating multiple models
together on a test dataset, and showcases how we can compare their performance. \
Those notebooks focus on BERT family of models fine-tuned for NER task. \
These models have advantages:
- They are smaller than LLMs like GPT-2 (largest tested model had 110M parameters)
- they come in many sizes, from small to large
- We can easily fine-tune them with custom tokens for a specific task
- They reach high accurancy after small number of epochs\

Sadly, they have disadvantages as well:
- We will need to fine-tune with different tokens and data for different tasks
- We will probably need to heavily preprocess data before fine-tuning to reach their maximum potential
- As a result we get words and their tokens; we will have to do potentially heavy post-processing to get the final results

In my personal opinion, these models will be a great choice if we decide to focus on a small number
of tasks on the client's smartphone. We could hold task-specific models on server and client would have to download them only once. \
This approach of course doesn't work if we eant to perform e.g. 50 different tasks. In that case we might not be able to escape
prompt-engineering with LLMS like LLamas family of models.