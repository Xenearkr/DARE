"""EvalPlus / Dream-Coder instruct prompt helpers (aligned with d3LLM evalplus)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# d3LLM evalplus/prepare_instruct_prompts.py + provider/utility.py (run_code_eval.sh dream_coder)
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the problem "
    "and passes corresponding tests:"
)

HUMANEVAL_LEGACY_PREFIX = "Complete the following python code:\n"
MBPP_TASK_RE = re.compile(
    r"here is your task:\s*(.*?)\s*Your code should pass these tests:",
    re.DOTALL | re.IGNORECASE,
)
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def format_evalplus_user_content(task_body: str) -> str:
    """User turn only (task fenced with plain ```, matching make_raw_chat_prompt)."""
    body = task_body.strip()
    return f"{INSTRUCTION_PREFIX}\n```\n{body}\n```"


def format_evalplus_messages(task_body: str, include_assistant_prefix: bool = True) -> List[Dict[str, str]]:
    """Chat messages for RL parquet (assistant prefix matches EvalPlus decoding)."""
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": format_evalplus_user_content(task_body)},
    ]
    if include_assistant_prefix:
        messages.append(
            {
                "role": "assistant",
                "content": f"{RESPONSE_PREFIX}\n```python\n",
            }
        )
    return messages


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def extract_task_body_from_row(row: Dict[str, Any]) -> Optional[str]:
    """Recover raw task text from an existing RL parquet row."""
    data_source = row.get("data_source", "")
    prompt = _as_list(row.get("prompt"))
    if not prompt:
        return None

    if data_source == "mbpp":
        for msg in reversed(prompt):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            m = MBPP_TASK_RE.search(content)
            if m:
                return m.group(1).strip()
        return None

    # Single-turn datasets
    content = prompt[-1].get("content", "") if prompt[-1].get("role") == "user" else prompt[0].get("content", "")
    if not content:
        return None

    if content.startswith(HUMANEVAL_LEGACY_PREFIX):
        return content[len(HUMANEVAL_LEGACY_PREFIX) :].strip()

    if data_source in {"humaneval", "humanevalplus"}:
        return content.strip()

    return content.strip()


def extract_taco_codeblock(content: str) -> Optional[str]:
    blocks = CODE_BLOCK_RE.findall(content)
    if not blocks:
        return None
    for block in blocks:
        if re.search(r"def\s+\w+\s*\(", block):
            return block.strip()
    return blocks[0].strip()


def convert_row_to_evalplus(row: Dict[str, Any], *, index: int, split: str) -> Optional[Dict[str, Any]]:
    """Convert one RL parquet row to EvalPlus prompt format; preserve reward fields."""
    data_source = row.get("data_source", "")
    task_body = extract_task_body_from_row(row)
    if not task_body:
        return None

    reward_model = row.get("reward_model")
    if hasattr(reward_model, "to_dict"):
        reward_model = reward_model.to_dict()
    extra_info = row.get("extra_info") or {}
    if hasattr(extra_info, "to_dict"):
        extra_info = extra_info.to_dict()

    new_row = {
        "data_source": data_source,
        "prompt": format_evalplus_messages(task_body, include_assistant_prefix=True),
        "reward_model": reward_model,
        "extra_info": dict(extra_info),
    }
    new_row["extra_info"]["split"] = split
    new_row["extra_info"]["index"] = index
    new_row["extra_info"]["prompt_style"] = "evalplus"
    return new_row
