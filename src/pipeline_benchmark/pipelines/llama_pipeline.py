import re
from typing import List
from tqdm import tqdm
from llama_cpp import Llama


N_CTX = 8192
MAX_TOKENS = 2048
SEED = 69


def run_llama_pipeline(input_data: List[str], model_path: str, prompt_type: str) -> List[str]:
    """
    Runs the Llama pipeline on a list of inputs.

    :param input_data: list of inputs to extract coupons from
    :param model_path: path to the gguf file
    :param prompt_type: type of the prompt, can be either "w" or "wth"
    :return: list of model responses
    """
    if prompt_type not in ["w", "wth"]:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    llama = Llama(model_path, n_gpu_layers=100, n_ctx=N_CTX)

    output = []
    for inp in tqdm(input_data, desc="Llama inference"):
        prompt = "### Input:\n" + inp + "\n\n### Response:\n"

        if prompt_type == "w":
            prompt = "You are provided with text representing contents of the phone screen. Your task is to extract information about coupons from the text. The information should include the product name, the validity text, the discount text and the activation text.\n\n" + prompt

        out = llama.create_completion(prompt, max_tokens=MAX_TOKENS, stop=["]"], seed=SEED)
        out_text = out["choices"][0]["text"]

        last_brace_idx = out_text.rfind("}")
        if last_brace_idx != -1:
            res = out_text[:last_brace_idx + 1] + "]"
        else:
            res = "[]"

        output.append(res)

    return output
