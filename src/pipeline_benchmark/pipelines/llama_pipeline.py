from typing import List
from tqdm import tqdm
from llama_cpp import Llama


N_CTX = 16384


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
    for input in tqdm(input_data, desc="Llama inference"):
        prompt = "### Input:\n" + input + "\n\n### Response:\n"

        if prompt_type == "w":
            prompt = "You are provided with text representing contents of the phone screen. Your task is to extract information about coupons from the text. The information should include the product name, the validity date, the discount, the old price, and the new price.\n\n" + prompt

        # It seems that llama has a tendency to correctly recognize that there is no coupons in the input
        # and print "[]", but after that start to generate garbage. That's why adding "[]" to stop sequences
        # provides a significant speedup.
        out = llama.create_completion(prompt, max_tokens=N_CTX - len(prompt), stop=["### Response:", "### Input:", "[]"])["choices"][0]["text"]

        if out == "":
            out = "[]"

        output.append(out)

    return output
