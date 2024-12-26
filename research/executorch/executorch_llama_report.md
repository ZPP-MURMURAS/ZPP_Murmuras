# Introduction

The [ExecuTorch repository](https://github.com/pytorch/executorch) provides examples of how to deploy Llama models on mobile devices. This guide offers a concise summary and highlights the most relevant parts of the repository for this project.


# Models

Based on my understanding, ExecuTorch allows for the deployment of any model that can be exported using `torch.export`. While deploying a large language model (LLM) can be challenging, the repository offers a complete framework for Llama deployment, including model export scripts and a demo application.


# Demo App

There is a [demo Llama application](https://github.com/pytorch/executorch/tree/main/examples/demo-apps/android/LlamaDemo) in the repository. Executorch supports various backends for running models, including [non-CPU backends](https://github.com/pytorch/executorch/blob/main/examples/models/llama/non_cpu_backends.md). I only focused on the CPU-based XNNPACK. I was able to build and run the application with Llama 3.2 1B QAT+LoRA model on Samsung Galaxy A25, using the instructions provided [here](https://github.com/pytorch/executorch/blob/main/examples/demo-apps/android/LlamaDemo/docs/delegates/xnnpack_README.md). The application also supports a multimodal text/vision model [LLaVA](https://huggingface.co/llava-hf/llava-1.5-7b-hf), however I was not able to run it, because the model generation script required more than 32 GB of RAM.


# Benchmarking

The repository includes a [benchmarking section](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md) where different Llama models are tested on edge devices using the XNNPACK backend. By following [these instructions](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md#instructions), I was able to run the benchmark with Llama 3.2 1B QAT+LoRA on Samsung Galaxy A25.


# Fine-tuning

Models can be fine-tuned using the [torchtune](https://github.com/pytorch/torchtune) library. Detailed instructions can be found [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama/UTILS.md#finetuning).


# Quantization
Models can be quantized using the [torchao](https://github.com/pytorch/ao) library.


# Structured Format Enforcement

I was not able to find any straightforward way of enforcing the Llama output to be in a structured format, such as JSON. One possible approach might involve using the [outlines](https://github.com/dottxt-ai/outlines) library and [exporting to ExecuTorch](https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html). However, setting everything up could be quite time-consuming, and it's unclear whether this approach would work as intended.


# Conclusions

- Llama deployment on edge devices is straightforward and effective with ExecuTorch. 
- Lack of structured format enforcement options could prove to be a problem. It remains to be seen, if fine-tuning the model is a sufficient solution.
- Benchmarking multimodal text/vision models would be interesting, but it is highly unlikely that they will be practical, because of their high parameter counts (e.g., 7B for LLaVA). However, as can be seen [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md#llama-331-8b), running quantized large models is not entirely out of the question on high-end phones.
