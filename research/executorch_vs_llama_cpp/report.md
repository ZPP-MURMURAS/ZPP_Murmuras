# Preface
This file summarizes my findings regarding Llama speed comparisons on a mobile phone.
I compared ExecuTorch and Llama.cpp frameworks, as well as, the impact of quantization.

The tests were performed on a Samsung Galaxy A25 with 6 GB of RAM and both frameworks utilized only the CPU.
For every test, I performed a warm-up run and measured the model's speed on 8 runs. 
If we are planning to process data in bulk, results after a warm-up should be representative of what can be expected.

Both Llama.cpp and ExecuTorch repositories contain example programs that allow us to 
run a selection of LLMs and see some basic statistics regarding the model's performance. For my
experiment I used the instructions described [here](https://github.com/ZPP-MURMURAS/ZPP_Murmuras/blob/main/research/llama_cpp/introduction/llama.ipynb)
for Llama.cpp and [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md)
for ExecuTorch.

I decided to use [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) as the base model for both the framework comparison and quantization tests.
I did not find a way to run models with the same quantization schemes on both
frameworks and larger unquantized models had RAM requirements, which my phone did not meet.

With ExecuTorch, I tested [SpinQuant](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8) 
and [QAT+LoRA](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8) quantizations.
You can find a similar benchmark [here](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md),
however the phones that where used in it are relatively high-end and for this project, we are interested in
a more common case.

With Llama.cpp, I tested Q8\_0 and Q4\_0 quantizations.
See [here](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md) for more details.

# Framework Comparison Results
|                                   | ExecuTorch  | Llama.cpp    |
|-----------------------------------|-------------|--------------|
| Total Speed (tokens/second)       | 4.48 ± 0.11 | 5.45 ± 0.14  |
| Prompt Eval Speed (tokens/second) | 7.14 ± 0.31 | 28.47 ± 4.29 |
| Generation Speed (tokens/second)  | 5.31 ± 0.18 | 5.64 ± 0.12  |

# ExecuTorch Quantization Results
|                                   | Baseline    | SpinQuant    | QAT+LoRA     |
|-----------------------------------|-------------|--------------|--------------|
| Total Speed (tokens/second)       | 4.48 ± 0.11 | 15.12 ± 0.82 | 15.03 ± 0.46 |
| Prompt Eval Speed (tokens/second) | 7.14 ± 0.31 | 57.72 ± 0.66 | 54.48 ± 1.02 |
| Generation Speed (tokens/second)  | 5.31 ± 0.18 | 15.42 ± 0.87 | 15.34 ± 0.48 |

# Llama.cpp Quantization Results
|                                   | Baseline     | Q8\_0       | Q4\_0        |
|-----------------------------------|--------------|-------------|--------------|
| Total Speed (tokens/second)       | 5.45 ± 0.14  | 7.55 ± 0.23 | 10.69 ± 0.06 |
| Prompt Eval Speed (tokens/second) | 28.47 ± 4.29 | 9.83 ± 0.21 | 31.52 ± 0.67 |
| Generation Speed (tokens/second)  | 5.64 ± 0.12  | 9.45 ± 0.34 | 11.58 ± 0.08 |

# Conclusions
Llama.cpp seems to outperform ExecuTorch. Properly evalueting quantizations would require model accuracy data.

# Note
I have also included my ad hoc scripts that I used for gathering the data. 
One benchmark uses adb, while the other needs to be run directly on device.
