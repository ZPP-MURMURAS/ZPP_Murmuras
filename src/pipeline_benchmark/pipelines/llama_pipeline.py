from typing import List
from tqdm import tqdm
from llama_cpp import Llama


def run_llama_pipeline(input_data: List[str], model_path: str) -> List[str]:
    llama = Llama(model_path, n_gpu_layers=100)
    output = []    

    for input in tqdm(input_data):
        prompt = "### Input:\n" + input + "\n\n### Response:\n"
        out = llama(prompt, max_tokens=1024, stop=["### Response:", "[]"])["choices"][0]["text"]
        if out == "":
            out = "[]"
        output.append(out)

    return output
