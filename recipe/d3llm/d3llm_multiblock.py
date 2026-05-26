"""
Bind d3LLM entropy-based multi-block generation onto a loaded DreamModel.

Does not modify verl training code. Requires d3LLM repo on PYTHONPATH or D3LLM_ROOT.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class D3LLMMultiBlockConfig:
    """Defaults aligned with d3LLM Dream-Coder eval (run_code_eval.sh)."""

    block_length: int = 32
    threshold: float = 0.5
    block_add_threshold: float = 0.1
    decoded_token_threshold: float = 0.95
    cache_delay_iter: int = 32
    early_stop: bool = True
    alg: str = "entropy_threshold"
    temperature: float = 0.0
    max_new_tokens: int = 256


def ensure_d3llm_on_path(d3llm_root: Optional[str] = None) -> Path:
    root = Path(d3llm_root or __import__("os").environ.get("D3LLM_ROOT", "/home/u-liujc/Codes/d3LLM")).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"d3LLM repo not found: {root}. Set D3LLM_ROOT.")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def bind_multiblock(model: Any, cfg: Optional[D3LLMMultiBlockConfig] = None, d3llm_root: Optional[str] = None) -> Any:
    """Attach generate_multi_block and multiblock-aware diffusion_generate to model."""
    ensure_d3llm_on_path(d3llm_root)
    from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationMixin

    cfg = cfg or D3LLMMultiBlockConfig()
    model.generate_multi_block = types.MethodType(DreamGenerationMixin.generate_multi_block, model)
    model._sample_multi_block = types.MethodType(DreamGenerationMixin._sample_multi_block, model)
    model._sample_multi_block_kv_cache = types.MethodType(
        DreamGenerationMixin._sample_multi_block_kv_cache, model
    )
    model._prepare_inputs = types.MethodType(DreamGenerationMixin._prepare_inputs, model)
    model.diffusion_generate = types.MethodType(_make_diffusion_generate(cfg), model)
    return model


def _make_diffusion_generate(cfg: D3LLMMultiBlockConfig):
    def diffusion_generate(
        model_self,
        input_ids,
        attention_mask=None,
        max_new_tokens=None,
        output_history=False,
        return_dict_in_generate=True,
        steps=None,
        temperature=None,
        top_p=None,
        alg=None,
        alg_temp=None,
        threshold=None,
        block_length=None,
        **kw,
    ):
        from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationConfig

        t = temperature if temperature is not None else cfg.temperature
        alg_ = alg or cfg.alg
        th = threshold if threshold is not None else cfg.threshold
        bl = block_length if block_length is not None else cfg.block_length
        gen_config = DreamGenerationConfig(
            max_length=input_ids.shape[1] + (max_new_tokens or cfg.max_new_tokens),
            mask_token_id=model_self.config.mask_token_id,
            temperature=t,
            alg=alg_,
            return_dict_in_generate=return_dict_in_generate,
        )
        result, nfe = model_self.generate_multi_block(
            inputs=input_ids,
            generation_config=gen_config,
            attention_mask=attention_mask,
            threshold=th,
            block_size=bl,
            block_add_threshold=cfg.block_add_threshold,
            decoded_token_threshold=cfg.decoded_token_threshold,
            cache_delay_iter=cfg.cache_delay_iter,
            early_stop=cfg.early_stop,
        )
        return result, nfe

    return diffusion_generate
