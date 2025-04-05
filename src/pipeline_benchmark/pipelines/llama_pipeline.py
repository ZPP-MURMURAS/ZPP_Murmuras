from typing import List
from llama_cpp import Llama


def run_llama_pipeline(input_data: List[str], model_path: str) -> List[str]:
    llama = Llama(model_path)
    output = []    

    for input in input_data:
      prompt = input + "\n\n### Response:\n"
      out = llama(prompt, max_tokens=512)["choices"][0]["text"]
      output.append(out)

    return output
