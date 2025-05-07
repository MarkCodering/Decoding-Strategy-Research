import os
import csv
import json
import random
import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime
import torch
import numpy as np
from typing import Dict, List, Optional, Literal, Tuple
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TextStreamer,
)
from transformers.generation.logits_process import LogitsProcessorList
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Local imports
from tcd import TCD

# Initialize logger
def setup_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=level,
    )

# DataClass for configuration
@dataclass
class EvalConfig:
    llm_model: Optional[str] = None
    vlm_model: Optional[str] = None
    dataset: Literal["mmlu", "mmlu-pro", "commonsenseqa"] = "commonsenseqa"
    use_tcd: bool = False
    tcd_penalty: float = 0.4
    tcd_temperature: float = 1.0
    max_new_tokens: int = 1
    seed: int = 42
    access_token: str = field(default="-1", repr=False)
    log_dir: str = "log"
    device: torch.device = field(init=False)

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utility functions
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()

# Model loading
def get_bits_and_bytes_config(tensor_type: str) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_8bit=(tensor_type == "int8"),
        load_in_4bit=(tensor_type in {"fp4", "nf4"}),
        bnb_4bit_quant_type=("nf4" if tensor_type == "nf4" else "fp4"),
        bnb_4bit_compute_dtype=torch.float16,
    )

def load_model(
    llm_path: Optional[str],
    vlm_path: Optional[str],
    token: str,
    tensor_type: Literal["fp16", "bf16", "int8", "fp4", "nf4"] = "fp16",
    device: torch.device = torch.device("cpu"),
) -> Tuple:
    config_kwargs = {"torch_dtype": {"fp16": torch.float16, "bf16": torch.bfloat16}.get(tensor_type, torch.float32)}
    if tensor_type in {"int8", "fp4", "nf4"}:
        config_kwargs["quantization_config"] = get_bits_and_bytes_config(tensor_type)

    if not llm_path and not vlm_path:
        raise ValueError("At least one of llm_path or vlm_path must be provided")

    tokenizer, model = None, None

    if llm_path:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path, trust_remote_code=True, use_auth_token=token
        )
        model = AutoModelForCausalLM.from_pretrained(
            llm_path, trust_remote_code=True, use_auth_token=token, **config_kwargs
        ).eval().to(device)
        logging.info(f"Loaded LLM from {llm_path}")

    if vlm_path:
        tokenizer = AutoProcessor.from_pretrained(
            vlm_path, trust_remote_code=True, use_auth_token=token
        )
        model = AutoModelForImageTextToText.from_pretrained(
            vlm_path, trust_remote_code=True, use_auth_token=token, **config_kwargs
        ).eval().to(device)
        logging.info(f"Loaded VLM from {vlm_path}")

    return tokenizer, model

# Dataset loading and preprocessing
def load_and_preprocess(
    name: Literal["mmlu", "mmlu-pro", "commonsenseqa"],
    has_noise: bool = True,
    is_prompting_fix: bool = False,
):
    def preprocess_mmlu(example):
        choices = ["A", "B", "C", "D"]
        sys_msg = f"Multiple choice questions about {example['subject']}.)"
        opts = "\n".join(f"{c}. {a}" for c, a in zip(choices, example['choices']))
        prompt = f"{sys_msg}\n{example['question']}\n{opts}\nAnswer: {' ' if has_noise else ''}"
        return {"question": prompt, "answer_str": choices[example['answer']]}  # map idx to letter

    def preprocess_mmlu_pro(example):
        choice_map = "ABCDEFGHIJ"            
        opts = "\n".join(f"{choice_map[i]}. {opt}" for i, opt in enumerate(example['options']))
        prompt = f"Multiple choice about {example['category']}: {example['question']}\nOptions:\n{opts}\nAnswer: {' ' if has_noise else ''}"
        return {"question": prompt, "answer_str": example['answer']}

    def preprocess_commonsenseqa(example):
        labels = ["A", "B", "C", "D", "E"]
        opts = "\n".join(f"{l}. {c}" for l, c in zip(labels, example['choices']['text']))
        prompt = f"Question: {example['question']}\n{opts}\nAnswer: {' ' if has_noise else ''}"
        return {"question": prompt}

    if name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test").map(preprocess_mmlu)
    elif name == "mmlu-pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").map(preprocess_mmlu_pro)
    elif name == "commonsenseqa":
        ds = load_dataset("tau/commonsense_qa", split="validation").map(preprocess_commonsenseqa)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return ds

# Prompt extraction

def extract_prompt_and_answer(
    sample: Dict,
    dataset: Literal["mmlu", "mmlu-pro", "commonsenseqa"],
) -> Tuple[str, str]:
    if dataset in {"mmlu", "mmlu-pro"}:
        return sample["answer_str"], sample["question"]
    if dataset == "commonsenseqa":
        return sample["answerKey"], sample["question"]
    raise ValueError("Invalid dataset")

# Metrics calculation
def calculate_metrics(
    truths: List[str], preds: List[str], allow_random: bool = False
) -> Dict[str, float]:
    cleaned_preds = []
    for p in preds:
        if not p and allow_random:
            cleaned_preds.append(random.choice(["A", "B", "C", "D", "E"]))
        else:
            cleaned_preds.append(p or "")
    return {
        "accuracy": accuracy_score(truths, cleaned_preds),
        "f1": f1_score(truths, cleaned_preds, average="weighted", zero_division=1),
        "precision": precision_score(truths, cleaned_preds, average="weighted", zero_division=1),
        "recall": recall_score(truths, cleaned_preds, average="weighted", zero_division=1),
    }

# Logging results
def save_results(log: Dict, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(log_dir, f"log_{timestamp}.json")
    with open(json_path, "w") as jf:
        json.dump(log, jf, indent=2)
    csv_path = os.path.join(log_dir, "results.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=list(log.keys()),
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(log)
    logging.info(f"Results saved to {json_path} and {csv_path}")

# Main evaluation
def evaluate(config: EvalConfig) -> None:
    set_random_seed(config.seed)
    tokenizer, model = load_model(
        llm_path=config.llm_model,
        vlm_path=config.vlm_model,
        token=config.access_token,
        device=config.device,
    )

    dataset = load_and_preprocess(config.dataset)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    logits_processors = None
    if config.use_tcd:
        tcd = TCD(
            allowed_tokens=["A", "B", "C", "D", "E"],
            tokenizer=tokenizer,
            penalty=config.tcd_penalty,
            temperature=config.tcd_temperature,
        )
        logits_processors = LogitsProcessorList([tcd])

    results = {"PREDICTION": [], "TRUTH": []}
    for sample in tqdm(dataset, desc="Evaluating"):
        truth, prompt = extract_prompt_and_answer(sample, config.dataset)
        inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            logits_processor=logits_processors,
        )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = pred_text[len(prompt) :].strip()
        results["PREDICTION"].append(pred)
        results["TRUTH"].append(truth)

    metrics = calculate_metrics(results["TRUTH"], results["PREDICTION"])
    logging.info(f"Metrics: {metrics}")

    log_data = {**metrics, "dataset": config.dataset, "model": config.llm_model or config.vlm_model}
    save_results(log_data, config.log_dir)

# CLI parser

def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate LLM/VLM models on MCQ datasets.")
    parser.add_argument("--llm_model", type=str, help="Path or name of the LLM model.")
    parser.add_argument("--vlm_model", type=str, help="Path or name of the VLM model.")
    parser.add_argument(
        "--dataset", choices=["mmlu", "mmlu-pro", "commonsenseqa"], default="commonsenseqa"
    )
    parser.add_argument("--use_tcd", action="store_true", help="Enable TCD decoding strategy")
    parser.add_argument("--tcd_penalty", type=float, default=0.4, help="Penalty for TCD")
    parser.add_argument("--tcd_temperature", type=float, default=1.0, help="Temperature for TCD")
    parser.add_argument("--max_new_tokens", type=int, default=1)
    args = parser.parse_args()
    return EvalConfig(
        llm_model=args.llm_model,
        vlm_model=args.vlm_model,
        dataset=args.dataset,
        use_tcd=args.use_tcd,
        tcd_penalty=args.tcd_penalty,
        tcd_temperature=args.tcd_temperature,
        max_new_tokens=args.max_new_tokens,
    )

if __name__ == "__main__":
    setup_logger()
    cfg = parse_args()
    evaluate(cfg)
