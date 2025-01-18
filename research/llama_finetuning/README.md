# Fine-tuning Llamas
This directory contains results of my experiments with fine-tuning Llama models on various tasks. The main goal of this research is to understand how to fine-tune Llama models (in this case, meta-llama/Llama-3.2-1B) with different approaches.

The first approach can be found in the "collab_fine_tuning.ipynb" notebook. It is based on [this](https://huggingface.co/blog/ImranzamanML/fine-tuning-1b-llama-32-a-comprehensive-article) tutorial. \
It's the most straight-forward approach; we load model, tokenizer and dataset, we preprocess dataset, and we train the model. \
The biggest difference between this and fine-tuning BERTs is the usage of the [Unsloth](https://github.com/unslothai/unsloth) library. \
Its purpose is to significantly speed up the training process by using custom kernels for the most computationally expensive operations. \
Sadly, it requires a GPU with CUDA support, which I don't have access to, so the notebook I provide is ready to be run on Google Collab (and I suggest you run it this way).

The second approach can be found in the "modal_fine_tuning.py" file. It utilizes the [Modal](https://modal.com/) framework for fine-tuning LLMs. \
It works by deploying specified code on the server, inside a docker container that we can define, and then running said code inside the container. \
It also utilizes Unsloth, and it is basically the same code as in the first approach, but it's annotated with Modal-specific decorators to make the code \
run on the server, with the specified image and on specific GPUs.

The last approach can be found in the "modal_axolotl_fine_tuning" subdirectory. It utilizes the [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) framework for fine-tuning LLMs. \
Base version is "codeless" - you specify your configs in the .yaml file, and then you utilize CLI commands to start the fune-tuning. \
This makes it difficult to run it with Modal, which is why Modal created an example project which I modified and included here (it has MIT license). \
I managed to run it with my own model and dataset (by "my", I mean other than the ones prepared in the example project).

Out of all three approaches, the second one is my favourite. Unsloth makes fine-tuning very fast, and even though in its free version, it limits us to only \
one GPU, it's enough for our purposes. On the other hand, Axolotl approach is the one I strongly dislike. In its default setting, it was using 2 80GB VRAM GPUs, \
and I was met with OutOfMemory errors. Interestingly enough, while using 3 GPUs only made the problem worse, using one GPU fixed the error. \
Still, Axolotl was working on the limit (60-75GB of VRAM usage), while the Unsloth approach was using only 11GB of VRAM, and was significantly faster. \
There is an option to use unsloth with Axolotl, but only for LLama models, and Axolotl docs are a joke. \
Finally, Google Collab approach is alright; thanks to Unsloth I was able to run it on the T4 GPUs, and it's easy to use, but modal usage feels more "concise".