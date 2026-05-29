"""EvalPlus / Dream-Coder instruct prompt helpers (aligned with d3LLM evalplus)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

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
MBPP_TASK_AND_TESTS_RE = re.compile(
    r"here is your task:\s*(.*?)\s*Your code should pass these tests:\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _mbpp_user_messages(prompt: Any) -> List[Dict[str, str]]:
    return [m for m in _as_list(prompt) if m.get("role") == "user"]


def extract_mbpp_task_and_tests(
    row: Dict[str, Any],
) -> Optional[Tuple[str, List[str]]]:
    """Parse task text and assertion lines from the **last** MBPP user turn."""
    users = _mbpp_user_messages(row.get("prompt"))
    if not users:
        return None

    content = users[-1].get("content", "")
    match = MBPP_TASK_AND_TESTS_RE.search(content)
    if match:
        task = match.group(1).strip()
        tests = [ln.strip() for ln in match.group(2).splitlines() if ln.strip()]
        if task and tests:
            return task, tests

    # Fallback: task from regex + tests from reward_model.ground_truth
    task_only = MBPP_TASK_RE.search(content)
    if not task_only:
        return None
    task = task_only.group(1).strip()
    reward_model = row.get("reward_model") or {}
    if hasattr(reward_model, "to_dict"):
        reward_model = reward_model.to_dict()
    gt = reward_model.get("ground_truth")
    if gt is None:
        return None
    if isinstance(gt, str):
        tests = json.loads(gt)
    else:
        tests = list(gt)
    if not task or not tests:
        return None
    return task, [str(t).strip() for t in tests]


def format_mbpp_evalplus_task_body(task: str, tests: List[str]) -> str:
    """EvalPlus-style body with task + assertions (function names / arity)."""
    task = task.strip()
    if task and not task.endswith((".", "!", "?")):
        task += "."
    tests_block = "\n".join(t.strip() for t in tests if t.strip())
    return f"{task}\n\nYour code should pass these tests:\n{tests_block}"


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
        parsed = extract_mbpp_task_and_tests(row)
        if parsed:
            task, _tests = parsed
            return task
        for msg in reversed(_mbpp_user_messages(prompt)):
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
    reward_model = row.get("reward_model")
    if hasattr(reward_model, "to_dict"):
        reward_model = reward_model.to_dict()
    extra_info = row.get("extra_info") or {}
    if hasattr(extra_info, "to_dict"):
        extra_info = extra_info.to_dict()

    entry_point = None
    if data_source == "mbpp":
        parsed = extract_mbpp_task_and_tests(row)
        if not parsed:
            return None
        task, tests = parsed
        task_body = format_mbpp_evalplus_task_body(task, tests)
        m = re.search(r"assert\s+(\w+)", tests[0])
        if m:
            entry_point = m.group(1)
    else:
        task_body = extract_task_body_from_row(row)
        if not task_body:
            return None

    new_row = {
        "data_source": data_source,
        "prompt": format_evalplus_messages(task_body, include_assistant_prefix=True),
        "reward_model": reward_model,
        "extra_info": dict(extra_info),
    }
    new_row["extra_info"]["split"] = split
    new_row["extra_info"]["index"] = index
    new_row["extra_info"]["prompt_style"] = "evalplus"
    if entry_point:
        new_row["extra_info"]["entry_point"] = entry_point
    return new_row
