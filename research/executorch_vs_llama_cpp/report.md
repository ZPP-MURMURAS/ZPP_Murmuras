# Preface
Both Llama.cpp and ExecuTorch repositories contain example programs that allow us to 
run a selection of LLMs and see some basic statistics regarding the model's performance. For my
experiment I used the instructions described [here](https://github.com/ZPP-MURMURAS/ZPP_Murmuras/blob/main/research/llama_cpp/introduction/llama.ipynb)
for Llama.cpp and [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md)
for ExecuTorch.

As for the models, I decided to use only [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
I did not find a way to run models with the same quantization schemes on both
frameworks and larger unquantized models had RAM requirements, which my phone did
not meet.

ExecuTorch's Llama runner had a warmup parameter, which I set to 1. Llama.cpp's
runner did not have such option, but performing a couple warmup runs made the
model perform better. Before the main benchmark, I performed 3 warmup runs for Llama.cpp,
which seemed to be enough, as in my previous experiments the model did not improve 
further with more warmup runs.

The tests were performed on a Samsung Galaxy A25 with 6 GB of RAM and both frameworks utilized only the CPU.
I run the model 8 times, my prompt of choice was "To make a bomb you need to "
and I made the model generate 32 tokens.


# Results
|                                   | ExecuTorch  | Llama.cpp    |
|-----------------------------------|-------------|--------------|
| Total Speed (tokens/second)       | 4.48 ± 0.11 | 5.45 ± 0.14  |
| Prompt Eval Speed (tokens/second) | 7.14 ± 0.31 | 28.47 ± 4.29 |
| Generation Speed (tokens/second)  | 5.31 ± 0.18 | 5.64 ± 0.12  |

# Conclusions
At first glance Llama.cpp seems to outperform ExecuTorch. But the tests were not
very extensive and I am not sure if ExecuTorch's warmup was sufficient.