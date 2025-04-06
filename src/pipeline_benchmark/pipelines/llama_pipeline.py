import os
from typing import List
from tqdm import tqdm
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


def _download_file(hf_repo_id: str, file_path: str, cache_dir: str) -> str:
    """Downloads a file from Hugging Face Hub if not already cached."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    
    os.makedirs(cache_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=hf_repo_id,
        filename=file_path,
        cache_dir=cache_dir,
        token=hf_token
    )


def run_llama_pipeline(input_data: List[str], hf_repo_id: str, gguf_path: str, prompt_type: str, cache_dir: str = "./models") -> List[str]:
    if prompt_type not in ["w", "wth"]:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
        
    model_path = _download_file(hf_repo_id, gguf_path, cache_dir)
    llama = Llama(model_path, n_gpu_layers=100)

    output = []
    for input in tqdm(input_data, desc="Llama inference"):
        prompt = "### Input:\n" + input + "\n\n### Response:\n"

        if prompt_type == "w":
            prompt = "You are provided with text representing contents of the phone screen. Your task is to extract information about coupons from the text. The information should include the product name, the validity date, the discount, the old price, and the new price.\n\n" + prompt

        # It seems that llama has a tendency to correctly recognize that there is no coupons in the input
        # and print "[]", but after that start to generate garbage. That's why adding "[]" to stop sequences
        # provides a significant speedup.
        out = llama.create_completion(prompt, max_tokens=512, stop=["### Response:", "[]"])["choices"][0]["text"]

        if out == "":
            out = "[]"

        output.append(out)

    return output
