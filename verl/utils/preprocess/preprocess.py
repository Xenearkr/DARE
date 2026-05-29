import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import json
import uuid
import random

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "recipe" / "d3llm") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "recipe" / "d3llm"))
from evalplus_prompt import format_evalplus_messages  # noqa: E402

seed = 42
random.seed(seed)


mbpp_few_shots = [
    {"role": "user", "content": "You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n",},
    {"role": "assistant", "content": "[BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)' \n[DONE] \n\n ",},
    
    {"role": "user", "content": "You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \nassert is_not_prime(10) == True \nassert is_not_prime(35) == True \n",},
    {"role": "assistant", "content": "[BEGIN]\n 'import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result' \n[DONE] \n\n ",},
    
    {"role": "user", "content": "You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n",},
    {"role": "assistant", "content": "[BEGIN]\n 'import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums' \n[DONE] \n\n ",},
]


def load_dataset(input_path: str, repeat: int = 1, sample_num: int = None) -> List[Dict[str, Any]]:
    df = pd.read_parquet(input_path)
    dataset = df.to_dict("records")
    if sample_num is not None:
        dataset = dataset[:sample_num]
    dataset = dataset * repeat
    return dataset


def process_single_example(example: Dict[str, Any], idx: int, data_source: str, split: str, task: str) -> Optional[Dict[str, Any]]:
    """Process a single data sample, return unified data structure (shared by jsonl/parquet)."""
    question = example["problem"]
    
    # Handle answer extraction for gsm8k dataset
    if data_source == "gsm8k" and split == "train":
        answer = example["solution"].split("\n#### ")[-1].strip()
        example["answer"] = answer
    
    answer = example.get("answer", example.get("tests", None))
    
    # Build prompt_messages
    prompt_messages: List[Dict[str, str]] = [{
        "role": "user",
        "content": question
    }]
    if data_source == "mbpp" or data_source == "kodcode":
        answer_list = json.loads(answer)
        if not answer_list:
            print(f"No answer found for example {idx}")
            return None
        tests = "\n".join(answer_list)
        prompt_messages = mbpp_few_shots + [
            {
                "role": "user",
                "content": f"You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:\n\n {tests}  \n",
            },
            {
                "role": "assistant",
                "content": "[BEGIN]\n",
            },
        ]
    elif data_source in ("humaneval", "humanevalplus"):
        prompt_messages = format_evalplus_messages(question, include_assistant_prefix=True)
    
    data = {
        "data_source": data_source,
        "prompt": prompt_messages,
        "reward_model": {  # Used in RewardManager.__call__()
            "style": "rule",
            "ground_truth": answer
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "task": task,
        }
    }
    metadata = example.get("metadata", None)
    if metadata is not None and metadata != {}:
        data["extra_info"]["metadata"] = metadata
    if data_source == "countdown":
        data["extra_info"]["numbers"] = example["numbers"]  # str
    elif data_source.startswith("sudoku"):
        data["extra_info"]["puzzle"] = example["puzzle"]
    
    return data


def preprocess_dataset(input_path, parquet_path, dataset_name, split, task, repeat=1, sample_num=None, max_file_size_gb=2):
    dataset = load_dataset(input_path, repeat, sample_num)
    
    chunk = []
    file_idx = 1
    max_file_size = max_file_size_gb * 1024 * 1024 * 1024
    acc_size = 0
    total_rows = 0
    for idx, example in enumerate(dataset):
        processed = process_single_example(example, idx, dataset_name, split, task)
        if processed is None:
            continue
        
        # Buffer Parquet chunk
        row_size = len(json.dumps(processed, ensure_ascii=False).encode("utf-8"))
        acc_size += row_size
        chunk.append(processed)
        if acc_size >= max_file_size:
            df = pd.DataFrame(chunk)
            out_path = f"{parquet_path.replace('.parquet', '')}_{file_idx}.parquet"
            df.to_parquet(out_path)
            print(f"Write {out_path}, rows: {len(df)}")
            total_rows += len(df)
            file_idx += 1
            chunk = []
            acc_size = 0
    if chunk:
        print(chunk[0])
        df = pd.DataFrame(chunk)
        out_path = f"{parquet_path.replace('.parquet', '')}_{file_idx}.parquet"
        df.to_parquet(out_path)
        print(f"Write {out_path}, rows: {len(df)}")
        total_rows += len(df)
    
    print("Total rows:", total_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/source", type=str, required=False, help="Input directory containing raw json files")
    parser.add_argument("--output_dir", default="data/preprocessed", type=str, required=False, help="Output directory for processed files")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g., gsm8k)")
    parser.add_argument("--split", type=str, required=True, help="Split name (e.g., train or test)")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat times for parquet mode")
    parser.add_argument("--sample_num", type=int, default=None, help="Sample number for parquet")
    parser.add_argument("--task", default="math", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(f"{args.output_dir}/{args.split}", exist_ok=True)
    
    input_path = f"{args.input_dir}/{args.split}/{args.dataset_name}.parquet"
    parquet_path = f"{args.output_dir}/{args.split}/{args.dataset_name}.parquet"
    if args.sample_num is not None:
        parquet_path = parquet_path.replace(".parquet", f"-n{args.sample_num}.parquet")
    if args.repeat > 1:
        parquet_path = parquet_path.replace(".parquet", f"-r{args.repeat}.parquet")
    
    preprocess_dataset(input_path, parquet_path, args.dataset_name, args.split, args.task, repeat=args.repeat, max_file_size_gb=1.5, sample_num=args.sample_num)