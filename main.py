import os
import csv
import json
import random
from datetime import datetime
from typing import Literal
import argparse

from huggingface_hub import login
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TextStreamer,
)

from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from datasets import load_dataset
from tqdm import tqdm

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Helper function
from tcd import TCD

# Huggingface login
access_token = "-1"
login(token=access_token)

# Cuda support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


# Random seed
def set_random_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Model pipline
def load_model(
    llm_model_or_path: str | None = None,
    vlm_model_or_path: str | None = None,
    tensor_type: Literal["fp16", "bf16", "int8", "fp4", "nf4"] = "fp16",
):
    def _get_model_config(tensor_type: str) -> dict:
        model_config = {}
        dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}
        model_config["torch_dtype"] = dtype_mapping.get(tensor_type, "auto")

        if tensor_type in {"int8", "fp4", "nf4"}:
            model_config["quantization_config"] = BitsAndBytesConfig(
                **{
                    "load_in_8bit": tensor_type == "int8",
                    "load_in_4bit": tensor_type in {"fp4", "nf4"},
                    "bnb_4bit_quant_type": "nf4" if tensor_type == "nf4" else "fp4",
                    "bnb_4bit_compute_dtype": torch.float16,
                }
            )

        return model_config

    model_config = _get_model_config(tensor_type)

    # Error checking
    if llm_model_or_path is None and vlm_model_or_path is None:
        raise ValueError("Model name cannot be None")

    # Load LLM
    if llm_model_or_path:  # Not ""
        tokenizer = AutoTokenizer.from_pretrained(
            llm_model_or_path,
            trust_remote_code=True,
            token=access_token,
        )

        model = AutoModelForCausalLM.from_pretrained(
            llm_model_or_path,
            trust_remote_code=True,
            token=access_token,
            **model_config,
        ).eval()

        model.to(device)

        print(f"Load LLM: {llm_model_or_path} successfully.")

        # token_id
        # if model.config.eos_token_id is None:
        #    model.config.eos_token_id = tokenizer.eos_token_id

        # if model.config.pad_token_id is None:
        #    model.config.pad_token_id = tokenizer.pad_token_id

    # Load VLM
    if vlm_model_or_path:  # Not ""
        """
        Note: before using Qwen2.5-VL, we advise you to build from source with command:
            - pip install git+https://github.com/huggingface/transformers accelerate

        """

        # _is_lmdeploy_model = "lmdeploy:" in vlm_model_or_path

        # Load model directly
        tokenizer = AutoProcessor.from_pretrained(
            vlm_model_or_path,
            trust_remote_code=True,
            token=access_token,
            **(
                {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28}
                if "Qwen" in vlm_model_or_path
                else {}
            ),
        )

        # model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")
        model = AutoModelForImageTextToText.from_pretrained(
            vlm_model_or_path,
            trust_remote_code=True,
            token=access_token,
            **model_config,
        ).eval()

        print(f"Load VLM: {vlm_model_or_path} successfully.")

    return tokenizer, model


# Preprocess
def load_data(dataset_name: Literal["mmlu", "mmlu-pro", "commonsenseqa"]):
    # VARIABLE
    CHOICE = -1

    def mmlu_preprocess(example):
        noise = " "

        CHOICE = ["A", "B", "C", "D"]  # Mutiple choice with four option
        sys_msg = f"The following are multiple choice questions (with answers) about {example['subject']}."
        new_question = (
            sys_msg
            + "\n"
            + example["question"]
            + "\n"
            + "\n".join(
                [
                    f"{choice}. {answer}"
                    for choice, answer in zip(CHOICE, example["choices"])
                ]
            )
            + "\nAnswer:"
            + noise
        )
        example["question"] = new_question
        example["answer_str"] = CHOICE[example["answer"]]  # Original: 1 -> B
        # example["image_url"] = None
        return example

    def load_mmlu_pro():
        # load dataset
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        test_df = dataset["test"]
        # test_df = _mmlu_pro_preprocess(test_df)
        # val_df = dataset["validation"]
        # val_df = _mmlu_pro_preprocess(val_df)
        # process
        return test_df

    """
    def mmlu_pro_format_example(example): # no CoT
        # Ref: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py, format_example()
        sys_msg = "The following are multiple choice questions (with answers) about {}."
        question = example["?"]
        options = example["?"]
        example = "Question: {}\nOptions: ".format(question)
        choice_map = "ABCDEFGHIJ"
        for i, opt in enumerate(options):
            example += "{}. {}\n".format(choice_map[i], opt)
            example += "Answer: "
        return example
    """

    def mmlu_pro_preprocess(example):
        noise = " "

        # Question
        sys_msg = "The following are multiple choice questions (with answers) about {}.\n".format(
            example["category"]
        )
        question = example["question"]
        options = example["options"]
        question = sys_msg + "Question: {}\nOptions: ".format(question)
        choice_map = "ABCDEFGHIJ"
        for i, opt in enumerate(options):
            question += "{}. {}\n".format(choice_map[i], opt)
        question += "Answer: "

        example["question"] = question

        # Answer
        example["answer_str"] = example["answer"]
        return example

    def commonsenseqa_preprocess(example):
        noise = " "

        # Label 列表
        labels = ["A", "B", "C", "D", "E"]

        # 原始問題文字
        q = example["question"]

        # 開始組出新的 prompt
        formatted = f"Please answer the question by generate one token only with the options - A,B,C,D,E: {q}\n"
        for label, choice in zip(labels, example["choices"]["text"]):
            formatted += f"{label}. {choice}\n"

        formatted += "Answer: " + noise

        # 把欄位改成我們後續呼叫 get_prompt 能取到的格式
        example["question"] = formatted
        return example

    # default: None
    if dataset_name == "mmlu":
        # Reference: https://github.com/openai/evals/blob/main/examples/mmlu.ipynb
        dataset = load_dataset("cais/mmlu", "all", split="test")
        return dataset.map(mmlu_preprocess)
    elif dataset_name == "mmlu-pro":
        # Reference: https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py
        test_df = load_mmlu_pro()
        return test_df.map(mmlu_pro_preprocess)
    elif dataset_name == "commonsenseqa":
        dataset = load_dataset("tau/commonsense_qa")
        dataset = dataset["validation"]
        return dataset.map(commonsenseqa_preprocess)
    else:
        # ERROR
        raise NotImplementedError()


def get_prompt(data, dataset_name: Literal["mmlu", "mmlu-pro", "commonsenseqa"]):
    if dataset_name == "mmlu":
        prompt = data["question"]
        ans = data["answer_str"]
        return ans, prompt
    elif dataset_name == "mmlu-pro":
        prompt = data["question"]
        ans = data["answer_str"]
        return ans, prompt
    elif dataset_name == "commonsenseqa":
        prompt = data["question"]
        ans = data["answerKey"]
        return ans, prompt
    else:
        # ERROR
        raise NotImplementedError()


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
            choices = ["A", "B", "C", "D"]  # mmlu
            # choices = ["A", "B", "C", "D", "E", "F", "G", "H",  "I",  "J"] # mmlu pro
            choices = ["A", "B", "C", "D"]  # commonsenseqa
            pred = random.choice(choices)
        elif pred == "":
            choices = ["ELSE"]
            print('NO MATCH "" EXIST')
        processed_predictions.append(pred)

    accuracy = accuracy_score(all_answers, processed_predictions)
    f1 = f1_score(
        all_answers, processed_predictions, average="weighted", zero_division=1
    )
    precision = precision_score(
        all_answers, processed_predictions, average="weighted", zero_division=1
    )
    recall = recall_score(
        all_answers, processed_predictions, average="weighted", zero_division=1
    )

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
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

    # Save into csv
    have_decode = log_data.get("param", {}).get("stcm")
    row = {
        "model": log_data.get("Model", ""),
        "dataset_name": log_data.get("dataset_name", ""),
        "use_stcm": bool(have_decode),
        "penalty": have_decode.get("penalty", "") if have_decode else "",
        "temperature": have_decode.get("temperature", "") if have_decode else "",
        "accuracy": log_data.get("accuracy", ""),
        "f1_score": log_data.get("f1_score", ""),
        "precision": log_data.get("precision", ""),
        "recall": log_data.get("recall", ""),
    }

    csv_filename = os.path.join(log_dir, "experiment.csv")
    file_exists = os.path.exists(csv_filename)

    # 寫入或追加 CSV
    with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "model",
            "dataset_name",
            "use_stcm",
            "penalty",
            "temperature",
            "accuracy",
            "f1_score",
            "precision",
            "recall",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Log saved to {log_filename} and {csv_filename}")


def main(
    llm_model_or_path, vlm_model_or_path, dataset_name, params, debug: bool = False
):
    ### Parameter ###
    # allow token for decoding strategies
    if dataset_name == "mmlu":
        allowed_tokens = ["A", "B", "C", "D"]  # mmlu
    elif dataset_name == "mmlu-pro":
        allowed_tokens = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]  # mmlu-pro
    elif dataset_name == "commonsenseqa":
        allowed_tokens = ["A", "B", "C", "D", "E"]  # commonsenseqa
    else:
        pass

    # Logger
    LOG = {}
    LOG["Model"] = llm_model_or_path if vlm_model_or_path is None else vlm_model_or_path
    LOG["dataset_name"] = dataset_name

    # Load dataset: mmlu, mmlu-pro, commonsenseqa
    dataset = load_data(dataset_name=dataset_name)

    # Load model
    tokenizer, model = load_model(
        llm_model_or_path=llm_model_or_path, vlm_model_or_path=vlm_model_or_path
    )

    # Logit processor
    logits_processor = None

    if params.get("stcm") is not None:
        penalty = params["stcm"]["penalty"]
        temperature = params["stcm"]["temperature"]
        stcm = TCD(
            allowed_tokens=allowed_tokens,
            tokenizer=tokenizer,
            penalty=penalty,
            temperature=temperature,
            debug_mode=debug,
        )

        logits_processor = LogitsProcessorList()
        logits_processor.append(stcm)

    # Print token faster
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Evaluation Loop
    result = {
        "PREDICTION": [],
        "ANSWER": [],
    }

    for data in tqdm(dataset, desc="Evaluating"):
        # Get: input token
        ans, prompt = get_prompt(data=data, dataset_name=dataset_name)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        if "token_type_ids" in inputs and model_name == "tiiuae/Falcon3-7B-Base":
            # Ref: https://huggingface.co/stabilityai/stablecode-instruct-alpha-3b/discussions/7
            del inputs["token_type_ids"]

        # Get: output token
        # Doc for .generate(): https://jaketae.github.io/study/gpt2/#setup
        MAX_TOKEN_SIZE = 1

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKEN_SIZE,
            pad_token_id=(
                tokenizer.eos_token_id if llm_model_or_path is not None else None
            ),  # vlm does not have pad_token_id setting
            logits_processor=logits_processor,
            # streamer=streamer # print the model prediction in real time
        )

        # Collect Answer
        if params.get("stcm") is not None:
            generated_text = stcm.generate()
            pred = generated_text[0].strip()
        else:
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[
                0
            ]
            pred = generated_text[len(prompt) :].strip()

        # Save round result:
        result["PREDICTION"].append(pred)
        result["ANSWER"].append(ans)

    # debug
    if debug and params["stcm"] is not None:
        LOG["debug_info"] = stcm.dump_debug()

    # Compute metrics
    metrics = calculate_metrics(
        result["ANSWER"], result["PREDICTION"], allow_random=False
    )
    print("Metrics:", metrics)

    # Save log:
    # LOG["PREDICTION"] = result["PREDICTION"]
    # LOG["ANSWER"] = result["ANSWER"]
    LOG["param"] = params

    LOG.update(metrics)

    save_log(LOG, "log")


def get_parse():
    parser = argparse.ArgumentParser(
        description="Run evaluation with customizable parameters"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        choices=[
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-7B",
            "meta-llama/Llama-3.2-1B",
            "huggyllama/llama-7b",
        ],
        help="Path or name of the model to evaluate",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="commonsenseqa",
        choices=["mmlu", "mmlu-pro"],
        help="Dataset name",
    )
    parser.add_argument(
        "--use_stcm", action="store_true", help="Enable STCM if flag is provided"
    )
    parser.add_argument(
        "--stcm_penalty", type=float, default=0.4, help="Penalty value for STCM"
    )
    parser.add_argument(
        "--stcm_temperature", type=float, default=1.0, help="Temperature value for STCM"
    )
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--max_token_size",
        type=int,
        default=1,
        help="Maximum token size for generation",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Collection:
    - https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d
    """
    # Random seed
    set_random_seed(42)

    ### Parameter setting ###
    params = {
        "stcm": {"penalty": 0.4, "temperature": 1.0},
    }

    # TBD: "google/gemma-3-4b-it"
    support_llm_list = [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-32B",
        "meta-llama/Llama-3.2-1B",
        "huggyllama/llama-7b",  # llama-3.1 version
    ]

    # Experiment
    debug_mode = True
    dataset_name = "commonsenseqa"
    llm_list = [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "huggyllama/llama-7b",
        "tiiuae/Falcon3-1B-Base",
        "tiiuae/Falcon3-3B-Base",
        "tiiuae/Falcon3-7B-Base",
    ]
    penalty_list = [0.0, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    temperature_list = [1.0]

    for model_name in llm_list:
        for penalty in penalty_list:
            for temp in temperature_list:
                _params = {
                    "stcm": {"penalty": penalty, "temperature": temp},
                    # "stcm": None,
                }
                main(
                    llm_model_or_path=model_name,
                    vlm_model_or_path=None,
                    dataset_name=dataset_name,
                    params=_params,
                    debug=debug_mode,
                )
