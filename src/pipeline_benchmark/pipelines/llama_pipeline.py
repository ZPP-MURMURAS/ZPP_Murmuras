from typing import List
from tqdm import tqdm
from llama_cpp import Llama


def run_llama_pipeline(input_data: List[str], model_path: str, prompt_type: str) -> List[str]:
    llama = Llama(model_path, n_gpu_layers=100)
    output = []    

    for input in tqdm(input_data):
        prompt = "### Input:\n" + input + "\n\n### Response:\n"

        if prompt_type == "w":
            prompt = "You are provided with text representing contents of the phone screen. Your task is to extract information about coupons from the text. The information should include the product name, the validity date, the discount, the old price, and the new price.\n\n" + prompt

        out = llama(prompt, max_tokens=1024, stop=["### Response:", "[]"])["choices"][0]["text"]

        if out == "":
            out = "[]"

        output.append(out)

    return output
