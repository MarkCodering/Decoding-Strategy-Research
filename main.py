import os
import json
import random
from datetime import datetime

from huggingface_hub import login
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from datasets import load_dataset
from tqdm import tqdm

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Helper function
from stcm import STCM

# Huggingface login
login(token="")

# Cuda support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def load_model(model_name: str):
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

    tensor_type = "fp16"  # 可依需求調整，例如 "bf16", "int8", "fp4", "nf4"
    model_config = _get_model_config(tensor_type)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **model_config)
    
    # token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Load {model_name} successfully.")
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

def main(model_name, params):
    # Logger
    LOG = {}
    
    # Load model
    tokenizer, model = load_model(model_name)
    model.to(device)
    
    LOG["Model"] = model_name
    
    # Logit processor
    if params["stcm"] is not None:
        penalty = params["stcm"]["penalty"]
        temperature = params["stcm"]["temperature"]
        stcm = STCM(allowed_tokens=["A", "B", "C", "D"], tokenizer=tokenizer, penalty=penalty, temperature=temperature)
    
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
    
    test_DEBUG = {
        "LOGIT_before": [],
        "LOGIT_after": [],
    }
    
    for data in tqdm(dataset, desc="Evaluating"):
        # Dependency for mmlu
        prompt = data["question"]   
        ans = data["answer_str"]
        
        # input token
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        # inputs = tokenizer(prompt, return_tensors="pt")
        inputs.to(device)
        
        # output token
        MAX_TOKEN_SIZE = 1
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=MAX_TOKEN_SIZE,  
            
            attention_mask=inputs.attention_mask,
            pad_token_id = tokenizer.pad_token_id,
            
            logits_processor=logits_processor,
            #streamer=streamer # print the model prediction in real time
        )
        
        # Collect Answer
        if params["stcm"] is not None:
            generated_text = stcm.generate()
            pred = generated_text[0].strip()
        else:
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            pred = generated_text[len(prompt):].strip()
        
        # Save round result
        result["PREDICTION"].append(pred)
        result["ANSWER"].append(ans)
        
        # TEST
        #inputs.to("cpu")
        #outputs.to("cpu")
        
        #ids = tokenizer.convert_tokens_to_ids("A")
        #test_DEBUG["LOGIT_before"].append(inputs["input_ids"][0][ids])
        #test_DEBUG["LOGIT_after"].append(outputs[ids])

    # Compute metrics
    metrics = calculate_metrics(result["ANSWER"], result["PREDICTION"], allow_random=False)
    print("Metrics:", metrics)
    
    # Save log:
    LOG["PREDICTION"] = result["PREDICTION"]
    #LOG["ANSWER"] = result["ANSWER"]
    #LOG["LOGIT_before"] = test_DEBUG["LOGIT_before"]
    #LOG["LOGIT_after"] = test_DEBUG["LOGIT_after"]
    LOG["param"] = params
    
    LOG.update(metrics)
    
    save_log(LOG, "log")
        
if __name__ == "__main__":
    """
        Collection:
        - https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d
    """
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
    
    # Qwen report: https://arxiv.org/pdf/2412.15115
    model_list = ["huggyllama/llama-7b", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B", "meta-llama/Llama-3.2-1B", "google/gemma-3-4b-it"]
    
    penalty_list = [0.0, 0.2, 0.4, 0.8, 1.0]
    temperature_list = [1.0, 0.95, 0.9]
    
    set_random_seed(42)
    for i in range(1, 5):
        for j in penalty_list:
            for k in temperature_list:
                params = {
                    "stcm": {"penalty": j, "temperature": k},
                }
                main(model_list[i], params=params)