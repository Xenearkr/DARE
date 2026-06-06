"""EvalPlus helpers aligned with d3LLM Dream-Coder eval (import from vendored evalplus)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

_DARE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_EVALPLUS_ROOT = Path(
    os.environ.get(
        "DARE_EVALPLUS_ROOT",
        "/home/u-liujc/Codes/d3LLM/utils/utils_DreamCoder/code_eval/evalplus",
    )
)


def _ensure_evalplus_path() -> Path:
    root = Path(os.environ.get("DARE_EVALPLUS_ROOT", _DEFAULT_EVALPLUS_ROOT))
    if not (root / "evalplus" / "sanitize.py").exists():
        alt = _DARE_ROOT / "third_party" / "evalplus"
        if (alt / "evalplus" / "sanitize.py").exists():
            root = alt
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def evalplus_eos_strings() -> List[str]:
    """Match ``DLLMDecoder`` chat-template EOS list (``dllm.py`` + ``utility.EOS``)."""
    _ensure_evalplus_path()
    from evalplus.provider.utility import EOS  # noqa: WPS433

    eos = list(EOS)
    eos.append("\n```\n")
    return eos


def truncate_output_at_evalplus_eos(text: str, eos_list: Optional[List[str]] = None) -> str:
    """Copied from d3LLM ``evalplus/provider/dllm.py`` (lines 279-286)."""
    if eos_list is None:
        eos_list = evalplus_eos_strings()
    min_index = 10000
    for eos in eos_list:
        if eos in text:
            min_index = min(min_index, text.index(eos))
    if min_index == 10000:
        return text.replace("\t", "    ")
    return text[:min_index].replace("\t", "    ")


def count_gen_tokens_like_evalplus_dllm(tokenizer, response_text: str) -> int:
    """Copied from d3LLM ``evalplus/provider/dllm.py`` (lines 289-290)."""
    truncated = truncate_output_at_evalplus_eos(response_text)
    return len(tokenizer.encode(truncated, add_special_tokens=False))


def extract_evalplus_code_for_scoring(impl: str, entrypoint: str) -> Optional[str]:
    """Match d3LLM ``dllm.py`` postprocess: EOS truncate then ``evalplus.sanitize``."""
    if not impl or not str(impl).strip():
        return None
    truncated = truncate_output_at_evalplus_eos(str(impl))
    return sanitize_evalplus_impl(truncated, entrypoint)


def sanitize_evalplus_impl(impl: str, entrypoint: str) -> Optional[str]:
    """Match ``evalplus/codegen.py`` chat-template path (``impl`` only, not direct completion)."""
    if not impl or not str(impl).strip():
        return None
    _ensure_evalplus_path()
    from evalplus.sanitize import sanitize  # noqa: WPS433

    sanitized = sanitize(str(impl).strip(), entrypoint=entrypoint)
    return sanitized if sanitized else None
