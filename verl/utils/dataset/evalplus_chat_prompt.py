"""EvalPlus / Dream-Coder instruct decoding prompts (byte-aligned with d3LLM evalplus).

Reference: d3LLM ``evalplus/provider/utility.py::make_raw_chat_prompt``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from transformers import PreTrainedTokenizer

EVALPLUS_MAGIC_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem "
    "in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the problem "
    "and passes corresponding tests:"
)


def _as_messages(messages: Any) -> List[Dict[str, str]]:
    if messages is None:
        return []
    if isinstance(messages, list):
        return messages
    if hasattr(messages, "tolist"):
        return messages.tolist()
    return list(messages)


def is_evalplus_prompt(
    messages: Sequence[Dict[str, str]],
    extra_info: Optional[dict] = None,
) -> bool:
    if extra_info:
        style = extra_info.get("prompt_style") if isinstance(extra_info, dict) else None
        if style == "evalplus":
            return True

    msgs = _as_messages(messages)
    if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
        return False
    users = [m for m in msgs if m.get("role") == "user"]
    if not users:
        return False
    user_content = users[-1].get("content", "")
    asst_content = msgs[-1].get("content", "")
    if not isinstance(user_content, str) or not isinstance(asst_content, str):
        return False
    return (
        INSTRUCTION_PREFIX in user_content
        and RESPONSE_PREFIX in asst_content
        and "```python" in asst_content
    )


def make_evalplus_raw_chat_prompt(
    task_prompt: str,
    tokenizer: PreTrainedTokenizer,
    *,
    instruction_prefix: str = INSTRUCTION_PREFIX,
    response_prefix: str = RESPONSE_PREFIX,
) -> str:
    if tokenizer.chat_template is None:
        return task_prompt

    user_turn = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
    assistant_turn = f"""\
{response_prefix}
```python
{EVALPLUS_MAGIC_SPLITTER}
```
"""
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": assistant_turn},
        ],
        tokenize=False,
    )
    return full.split(EVALPLUS_MAGIC_SPLITTER)[0]


def extract_task_body_from_user_content(user_content: str) -> str:
    if "```\n" in user_content:
        return user_content.split("```\n", 1)[1].rsplit("\n```", 1)[0].strip()
    return user_content.strip()


def make_evalplus_raw_chat_prompt_from_messages(
    messages: Sequence[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
) -> str:
    msgs = _as_messages(messages)
    users = [m for m in msgs if m.get("role") == "user"]
    if not users:
        raise ValueError("EvalPlus messages require at least one user turn")
    task_body = extract_task_body_from_user_content(users[-1]["content"])
    return make_evalplus_raw_chat_prompt(task_body, tokenizer)


def render_decoding_prompt(
    messages: Sequence[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    extra_info: Optional[dict] = None,
    *,
    add_generation_prompt: bool = True,
) -> str:
    if is_evalplus_prompt(messages, extra_info):
        return make_evalplus_raw_chat_prompt_from_messages(messages, tokenizer)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def prompt_token_length(
    messages: Sequence[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    extra_info: Optional[dict] = None,
) -> int:
    text = render_decoding_prompt(messages, tokenizer, extra_info)
    return len(tokenizer.encode(text, add_special_tokens=False))
