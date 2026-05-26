# Copyright 2025 Shanghai AI Lab Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
d3LLM Dream-Coder multi-block decoding for DARE Dream HF rollout.

Bind logic follows ``recipe/d3llm/d3llm_multiblock.py`` and d3LLM
``eval_scripts/run_code_eval.sh`` (Dream-Coder branch: entropy_threshold,
block_length=32, cache_delay_iter=32, early_stop=True).

Generation core is vendored from d3LLM ``d3llm/d3llm_DREAM/d3llm_dream_generate_util.py``.
"""

from __future__ import annotations

import time
import types
from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
import torch.distributed as dist

from verl.workers.rollout.d3llm_dream_generate_util import DreamGenerationConfig, DreamGenerationMixin


@dataclass
class DreamMultiBlockConfig:
    """Defaults aligned with d3LLM ``run_code_eval.sh`` (d3llm_dream_coder)."""

    block_length: int = 32
    threshold: float = 0.5
    block_add_threshold: float = 0.1
    decoded_token_threshold: float = 0.95
    cache_delay_iter: int = 32
    early_stop: bool = True
    alg: str = "entropy_threshold"
    temperature: float = 0.0
    max_new_tokens: int = 256


def bind_multiblock(model: Any, cfg: DreamMultiBlockConfig | None = None) -> Any:
    """Attach d3LLM multi-block generation methods to a loaded Dream model."""
    cfg = cfg or DreamMultiBlockConfig()
    model.generate_multi_block = types.MethodType(DreamGenerationMixin.generate_multi_block, model)
    model._sample_multi_block = types.MethodType(DreamGenerationMixin._sample_multi_block, model)
    model._sample_multi_block_kv_cache = types.MethodType(
        DreamGenerationMixin._sample_multi_block_kv_cache, model
    )
    model._prepare_inputs = types.MethodType(DreamGenerationMixin._prepare_inputs, model)
    model.diffusion_generate = types.MethodType(_make_diffusion_generate(cfg), model)
    return model


def _make_diffusion_generate(cfg: DreamMultiBlockConfig):
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


def _extract_sequences(outputs) -> torch.Tensor:
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if hasattr(outputs, "sequences"):
        return outputs.sequences
    return outputs


def _pad_sequences_to_length(
    sequences: torch.Tensor,
    target_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    if sequences.size(1) >= target_len:
        return sequences[:, :target_len]
    pad = torch.full(
        (sequences.size(0), target_len - sequences.size(1)),
        pad_token_id,
        device=sequences.device,
        dtype=sequences.dtype,
    )
    return torch.cat([sequences, pad], dim=1)


def execute_dream_multiblock_generation(
    module,
    gen_kwargs: dict,
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
    process_outputs_fn,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """Run d3LLM multi-block rollout and delegate output parsing to ``process_outputs_fn``."""
    gen_length = gen_kwargs["gen_length"]
    block_length = gen_kwargs["block_length"]
    temperature = gen_kwargs["temperature"]
    do_sample = gen_kwargs.get("do_sample", False)
    per_sample_seed = gen_kwargs.get("per_sample_seed", True)
    prompt_length = idx_repeat.size(1)
    pad_token_id = gen_kwargs.get("pad_token_id", module.config.pad_token_id)

    mb_cfg = DreamMultiBlockConfig(
        block_length=block_length,
        threshold=gen_kwargs.get("threshold", 0.5),
        block_add_threshold=gen_kwargs.get("block_add_threshold", 0.1),
        decoded_token_threshold=gen_kwargs.get("decoded_token_threshold", 0.95),
        cache_delay_iter=gen_kwargs.get("cache_delay_iter", 32),
        early_stop=gen_kwargs.get("early_stop", True),
        max_new_tokens=gen_length,
        temperature=temperature,
    )
    bind_multiblock(module, cfg=mb_cfg)

    batch_size = idx_repeat.size(0)
    batch_start_time = time.time()
    top_p = 0.95 if (do_sample and temperature > 0) else (1.0 if temperature <= 0 else 0.95)
    target_len = prompt_length + response_length

    if per_sample_seed and batch_size > 1:
        seq_list = []
        for i in range(batch_size):
            if do_sample and temperature > 0:
                torch.manual_seed(int(gen_kwargs.get("base_seed", 42)) + i)
            out, _nfe = module.diffusion_generate(
                input_ids=idx_repeat[i : i + 1],
                attention_mask=attention_mask_repeat[i : i + 1] if attention_mask_repeat is not None else None,
                max_new_tokens=gen_length,
                temperature=temperature,
                top_p=top_p,
                threshold=mb_cfg.threshold,
                block_length=block_length,
                return_dict_in_generate=True,
            )
            seq = _pad_sequences_to_length(_extract_sequences(out), target_len, pad_token_id)
            seq_list.append(seq)
        outputs = torch.cat(seq_list, dim=0)
    else:
        out, _nfe = module.diffusion_generate(
            input_ids=idx_repeat,
            attention_mask=attention_mask_repeat,
            max_new_tokens=gen_length,
            temperature=temperature,
            top_p=top_p,
            threshold=mb_cfg.threshold,
            block_length=block_length,
            return_dict_in_generate=True,
        )
        outputs = _pad_sequences_to_length(_extract_sequences(out), target_len, pad_token_id)

    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    print(
        f"[RANK{rank}] Dream multi-block generation for {batch_size} samples cost: "
        f"{(time.time() - batch_start_time):.2f}s",
        flush=True,
    )

    return process_outputs_fn(
        outputs, idx_repeat, attention_mask_repeat, response_length, tokenizer, module.device
    )
