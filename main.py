import os
import json
import random
from datetime import datetime
from typing import Literal

from huggingface_hub import login
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoModelForImageTextToText,
    BitsAndBytesConfig, 
    TextStreamer
)

from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from datasets import load_dataset
from tqdm import tqdm

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Helper function
from stcm import STCM

# Huggingface login
access_token = "PUT_YOUR_TOKEN_HERE"
login(token=access_token)

# Cuda support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Random seed
def set_random_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Model pipline
def load_model(llm_model_or_path: str | None = None, vlm_model_or_path: str | None = None, tensor_type: Literal["fp16", "bf16", "int8", "fp4", "nf4"] = "fp16"):
    def _get_model_config(tensor_type: str) -> dict:
        model_config = {}
        dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}
        model_config["torch_dtype"] = dtype_mapping.get(tensor_type, "auto")

        if tensor_type in {"int8", "fp4", "nf4"}:
            model_config["quantization_config"] = BitsAndBytesConfig(**{
                "load_in_8bit": tensor_type == "int8",
                "load_in_4bit": tensor_type in {"fp4", "nf4"},
                "bnb_4bit_quant_type": "nf4" if tensor_type == "nf4" else "fp4",
                "bnb_4bit_compute_dtype": torch.float16,
            })

        return model_config

    model_config = _get_model_config(tensor_type)

    # Error checking
    if llm_model_or_path is None and vlm_model_or_path is None:
        raise ValueError("Model name cannot be None")
    
    # Load LLM
    if llm_model_or_path: # Not ""
        tokenizer = AutoTokenizer.from_pretrained(
            llm_model_or_path, trust_remote_code=True, token=access_token
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_or_path, trust_remote_code=True, token=access_token, **model_config
        ).eval()
        
        model.to(device)
        
        print(f"Load LLM: {llm_model_or_path} successfully.")
        
        # token_id
        #if model.config.eos_token_id is None:
        #    model.config.eos_token_id = tokenizer.eos_token_id
        
        #if model.config.pad_token_id is None:
        #    model.config.pad_token_id = tokenizer.pad_token_id
        
    # Load VLM
    if vlm_model_or_path: # Not ""
        """
        Note: before using Qwen2.5-VL, we advise you to build from source with command:
            - pip install git+https://github.com/huggingface/transformers accelerate

        """
        
        #_is_lmdeploy_model = "lmdeploy:" in vlm_model_or_path
        
        # Load model directly
        tokenizer = AutoProcessor.from_pretrained(
            vlm_model_or_path, trust_remote_code=True, token=access_token, 
            **(
                {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28}
                if "Qwen" in vlm_model_or_path else {}
            )
        )

        #model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")
        model = AutoModelForImageTextToText.from_pretrained(
            vlm_model_or_path, trust_remote_code=True, token=access_token, **model_config,
        ).eval()
        
        print(f"Load VLM: {vlm_model_or_path} successfully.")
    
    return tokenizer, model

# Preprocess
def preprocess(dataset):
    # mmlu
    # Reference: https://github.com/openai/evals/blob/main/examples/mmlu.ipynb
    def mmlu_preprocess(example):
        choices = ["A", "B", "C", "D"] # Mutiple choice with four option
        sys_msg = f"The following are multiple choice questions (with answers) about {example['subject']}."
        new_question = (
            sys_msg + "\n" + 
            example['question'] + "\n" +
            "\n".join([f"{choice}. {answer}" for choice, answer in zip(choices, example['choices'])]) +
            "\nAnswer:"
        )
        example["question"] = new_question
        example["answer_str"] = choices[example["answer"]] # Original: 1 -> B
        #example["image_url"] = None
        return example
    
    # default: mmlu_preprocess
    return dataset.map(mmlu_preprocess)

# Metrics
def calculate_metrics(
    all_answers: list,
    predictions: list,
    allow_random: bool = False,
) -> dict:
    """計算並回傳 accuracy, f1, precision 與 recall"""
    processed_predictions = []
    for pred in predictions:
        # 若預測結果為空字串且允許隨機則補上隨機選項
        if pred == "" and allow_random:
            choices = ["A", "B", "C", "D"]
            pred = random.choice(choices)
        elif pred == "":
            choices = ["ELSE"]
            print("NO MATCH \"\" EXIST")
        processed_predictions.append(pred)
    
    accuracy = accuracy_score(all_answers, processed_predictions)
    f1 = f1_score(all_answers, processed_predictions, average="weighted", zero_division=1)
    precision = precision_score(all_answers, processed_predictions, average="weighted", zero_division=1)
    recall = recall_score(all_answers, processed_predictions, average="weighted", zero_division=1)
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

def save_log(log_data: list, log_dir: str):
    # If log_dir is not exit, than create
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Save log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"log_{timestamp}.json")
    
    with open(log_filename, "w") as f:
        json.dump(log_data, f, indent=4)
    
    print(f"Log saved to {log_filename}")

def main(llm_model_or_path, vlm_model_or_path, params, debug: bool=False):
    ### Parameter ###
    allowed_tokens = ["A", "B", "C", "D"]
    
    # Logger
    LOG = {}
    
    # Load model   
    tokenizer, model = load_model(llm_model_or_path=llm_model_or_path, vlm_model_or_path=vlm_model_or_path)
    
    LOG["Model"] = llm_model_or_path if vlm_model_or_path is None else vlm_model_or_path
    
    # Logit processor
    logits_processor = None
    
    if params.get("stcm") is not None:
        penalty = params["stcm"]["penalty"]
        temperature = params["stcm"]["temperature"]
        stcm = STCM(allowed_tokens=allowed_tokens, tokenizer=tokenizer, penalty=penalty, temperature=temperature, debug_mode=debug)
    
        logits_processor = LogitsProcessorList()
        logits_processor.append(stcm)
    
    # Print token faster
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Load dataset: mmlu
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = preprocess(dataset)
    
    # Evaluation Loop
    result = {
        "PREDICTION": [],
        "ANSWER": [],
    }

    for data in tqdm(dataset, desc="Evaluating"):
        # Dependency for mmlu
        prompt = data["question"]   
        ans = data["answer_str"]
        
        # Get: input token
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Get: output token
        # Doc for .generate(): https://jaketae.github.io/study/gpt2/#setup
        MAX_TOKEN_SIZE = 1
        
        outputs = model.generate(
            **inputs,
            
            max_new_tokens=MAX_TOKEN_SIZE,  
            pad_token_id = tokenizer.eos_token_id if llm_model_or_path is not None else None, # vlm does not have pad_token_id setting
            
            logits_processor=logits_processor,
            #streamer=streamer # print the model prediction in real time
        )
        
        # Collect Answer
        if params.get("stcm") is not None:
            generated_text = stcm.generate()
            pred = generated_text[0].strip()
        else:
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            pred = generated_text[len(prompt):].strip()
        
        # Save round result
        result["PREDICTION"].append(pred)
        result["ANSWER"].append(ans)
        
    # debug
    if debug:
        LOG["debug_info"] = stcm.dump_debug()

    # Compute metrics
    metrics = calculate_metrics(result["ANSWER"], result["PREDICTION"], allow_random=False)
    print("Metrics:", metrics)
    
    # Save log:
    #LOG["PREDICTION"] = result["PREDICTION"]
    #LOG["ANSWER"] = result["ANSWER"]
    LOG["param"] = params
    
    LOG.update(metrics)
    
    save_log(LOG, "log")
        
if __name__ == "__main__":
    """
        Collection:
        - https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d
    """
    # Random seed
    set_random_seed(42)
    
    # TEST
    strategy_params = {
        "greedy": {},
        "contrastive": {"penalty_alpha": 0.6, "top_k": 4},
        "sampling": {"do_sample": True, "num_beams": 1},
        "beam_search": {"num_beams": 5},
        "beam_search_sampling": {"do_sample": True, "num_beams": 5},
        "diverse_beam_search": {
            "do_sample": False, "num_beams": 5, "num_beam_groups": 5, "diversity_penalty": 1.0
        },
        "self_speculative": {"do_sample": False, "assistant_early_exit": 4},
        "dola_high": {"do_sample": False, "dola_layers": "high"},
        "dola_low": {"do_sample": False, "dola_layers": "low"}
    }
    
    ### Parameter setting ###
    params = {
        "stcm": {"penalty": 0.0, "temperature": 1.0},
    }
    debug_mode = True
    
    # TBD: "google/gemma-3-4b-it"
    support_llm_list = [
        "Qwen/Qwen2.5-0.5B", 
        "Qwen/Qwen2.5-1.5B", 
        "Qwen/Qwen2.5-3B", 
        "Qwen/Qwen2.5-7B", 
        "Qwen/Qwen2.5-14B", 
        "Qwen/Qwen2.5-32B", 
        "meta-llama/Llama-3.2-1B", 
        "huggyllama/llama-7b" # llama-3.1 version
    ]
    support_vlm_list = [
        "Qwen/Qwen2-VL-7B-Instruct", 
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "liuhaotian/llava-v1.5-7b",
    ]
    
    # TEST: LLM
    #main(llm_model_or_path="Qwen/Qwen2.5-0.5B", vlm_model_or_path=None, params=params, debug=debug_mode)
    #main(llm_model_or_path="meta-llama/Llama-3.2-1B", vlm_model_or_path=None, params=params, debug=debug_mode)
    #main(llm_model_or_path="google/gemma-3-4b-it", vlm_model_or_path=None, params=params, debug=debug_mode) # Not support
    
    # TEST: VLM (BUG)
    #main(llm_model_or_path=None, vlm_model_or_path="Qwen/Qwen2.5-VL-3B-Instruct", params={}, debug=False)
    #main(llm_model_or_path=None, vlm_model_or_path="Qwen/Qwen2.5-VL-3B-Instruct", params=params, debug=True)
    #main(llm_model_or_path=None, vlm_model_or_path="Qwen/Qwen2.5-VL-3B-Instruct", params=params, debug=False)
    
    
    # Experiment
    llm_list = ["meta-llama/Llama-3.2-1B", "huggyllama/llama-7b"]
    penalty_list = [0.0, 0.4, 0.8, 1.0]
    temperature_list = [1.0, 0.95]
    for model_name in llm_list:
        for penalty in penalty_list:
            for temp in temperature_list:
                params = {
                    "stcm": {"penalty": penalty, "temperature": temp},
                }
                main(llm_model_or_path=model_name, vlm_model_or_path=None, params=params, debug=debug_mode)