#!/usr/bin/env python3
"""Verify DARE EvalPlus parquets match d3LLM make_raw_chat_prompt byte-for-byte."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from transformers import AutoTokenizer

DARE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DARE_ROOT))
sys.path.insert(0, str(DARE_ROOT / "recipe" / "d3llm"))
EVALPLUS_ROOT = Path("/home/u-liujc/Codes/d3LLM/utils/utils_DreamCoder/code_eval/evalplus")
sys.path.insert(0, str(EVALPLUS_ROOT))

from evalplus_prompt import extract_task_body_from_row, make_evalplus_raw_chat_prompt  # noqa: E402
from evalplus.provider.utility import make_raw_chat_prompt  # noqa: E402
from evalplus.data import get_human_eval_plus, get_mbpp_plus  # noqa: E402


INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the problem "
    "and passes corresponding tests:"
)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def expected_prompt(task_prompt: str, tokenizer) -> str:
    return make_raw_chat_prompt(
        task_prompt.strip(),
        INSTRUCTION_PREFIX,
        RESPONSE_PREFIX,
        tokenizer,
    )


def check_parquet(
    parquet_path: Path,
    official_tasks: Dict[str, Dict[str, Any]],
    *,
    tokenizer,
    dataset: str,
) -> Tuple[int, int, List[str]]:
    df = pd.read_parquet(parquet_path)
    mismatches: List[str] = []
    ok = 0
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        extra = row_dict.get("extra_info") or {}
        if hasattr(extra, "to_dict"):
            extra = extra.to_dict()
        task_id = extra.get("evalplus_task_id")
        if not task_id or task_id not in official_tasks:
            mismatches.append(f"missing evalplus_task_id or unknown task: {task_id}")
            continue
        official_prompt = official_tasks[task_id]["prompt"].strip()
        dare_task = extract_task_body_from_row(row_dict)
        if dare_task != official_prompt:
            mismatches.append(f"{task_id}: task body mismatch")
            continue
        prompt = _as_list(row_dict.get("prompt"))
        dare_decoded = make_evalplus_raw_chat_prompt(dare_task, tokenizer)
        ref = expected_prompt(official_prompt, tokenizer)
        if dare_decoded != ref:
            mismatches.append(f"{task_id}: decoding prompt mismatch ({len(dare_decoded)} vs {len(ref)} chars)")
            continue
        ok += 1
    return ok, len(df), mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/finetune_d3LLM")
    parser.add_argument(
        "--humaneval-parquet",
        default=str(DARE_ROOT / "data/preprocessed/rl/test/humaneval_evalplus_1.parquet"),
    )
    parser.add_argument(
        "--mbpp-parquet",
        default=str(DARE_ROOT / "data/preprocessed/rl/test/mbpp_evalplus_1.parquet"),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )

    he_ok, he_total, he_bad = check_parquet(
        Path(args.humaneval_parquet),
        get_human_eval_plus(),
        tokenizer=tokenizer,
        dataset="humaneval",
    )
    print(f"HumanEval: {he_ok}/{he_total} match")
    if he_bad:
        print("  first mismatches:", he_bad[:5])

    mbpp_ok, mbpp_total, mbpp_bad = check_parquet(
        Path(args.mbpp_parquet),
        get_mbpp_plus(),
        tokenizer=tokenizer,
        dataset="mbpp",
    )
    print(f"MBPP+: {mbpp_ok}/{mbpp_total} match")
    if mbpp_bad:
        print("  first mismatches:", mbpp_bad[:5])

    if he_ok != he_total or mbpp_ok != mbpp_total:
        raise SystemExit(1)
    print("All prompts match d3LLM make_raw_chat_prompt.")


if __name__ == "__main__":
    main()
