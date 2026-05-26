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

Training batches use left-padded prompts; d3LLM eval uses natural-length tokenization.
We strip left padding before generation and restore the left-padded layout for downstream
BGPO / FSDP training code.
"""

from __future__ import annotations

import time
import types
from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
import torch.distributed as dist

from verl.workers.rollout.d3llm_dream_generate_util import (
    DreamGenerationConfig,
    DreamGenerationMixin,
    set_fsdp_rollout_sync,
)
from verl.utils.fsdp_utils import fsdp_version
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
    if getattr(model, "_d3llm_multiblock_cfg", None) == cfg:
        return model
    model.generate_multi_block = types.MethodType(DreamGenerationMixin.generate_multi_block, model)
    model._sample_multi_block = types.MethodType(DreamGenerationMixin._sample_multi_block, model)
    model._sample_multi_block_kv_cache = types.MethodType(
        DreamGenerationMixin._sample_multi_block_kv_cache, model
    )
    model._prepare_inputs = types.MethodType(DreamGenerationMixin._prepare_inputs, model)
    model.diffusion_generate = types.MethodType(_make_diffusion_generate(cfg), model)
    model._d3llm_multiblock_cfg = cfg
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
        eos_token_id = getattr(model_self.generation_config, "eos_token_id", None)
        gen_config = DreamGenerationConfig(
            max_length=input_ids.shape[1] + (max_new_tokens or cfg.max_new_tokens),
            mask_token_id=model_self.config.mask_token_id,
            temperature=t,
            alg=alg_,
            return_dict_in_generate=return_dict_in_generate,
            eos_token_id=eos_token_id,
        )
        max_nfe = kw.pop("max_nfe", None)
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
            max_nfe=max_nfe,
        )
        return result, nfe

    return diffusion_generate


def _extract_sequences(outputs) -> torch.Tensor:
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if hasattr(outputs, "sequences"):
        return outputs.sequences
    return outputs


def _strip_left_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Convert left-padded batch to compact right-aligned sequences (d3LLM eval style)."""
    batch_size, width = input_ids.shape
    if attention_mask is None:
        lengths = torch.full((batch_size,), width, device=input_ids.device, dtype=torch.long)
        return input_ids, None, lengths

    lengths = attention_mask.sum(dim=1).long()
    max_len = int(lengths.max().item())
    if max_len == width:
        return input_ids, attention_mask, lengths

    compact_ids = torch.full(
        (batch_size, max_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device
    )
    compact_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    for i in range(batch_size):
        seq_len = int(lengths[i].item())
        if seq_len == 0:
            continue
        compact_ids[i, :seq_len] = input_ids[i, width - seq_len :]
        compact_mask[i, :seq_len] = 1
    return compact_ids, compact_mask, lengths


def _restore_padded_outputs(
    compact_outputs: torch.Tensor,
    prompt_lengths: torch.Tensor,
    original_prompt_width: int,
    response_length: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Map compact [prompt+response] back to left-padded [prompt_pad | response]."""
    batch_size = compact_outputs.size(0)
    target_len = original_prompt_width + response_length
    restored = torch.full(
        (batch_size, target_len),
        pad_token_id,
        dtype=compact_outputs.dtype,
        device=compact_outputs.device,
    )
    for i in range(batch_size):
        pl = int(prompt_lengths[i].item())
        prompt_part = compact_outputs[i, :pl]
        resp_part = compact_outputs[i, pl : pl + response_length]
        restored[i, original_prompt_width - pl : original_prompt_width] = prompt_part
        if resp_part.numel() > 0:
            n = min(resp_part.numel(), response_length)
            restored[i, original_prompt_width : original_prompt_width + n] = resp_part[:n]
    return restored


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


def _estimate_max_nfe(gen_length: int, block_length: int) -> int:
    """Upper bound for multiblock KV decode; avoids runaway loops on bad states."""
    num_blocks = (gen_length + block_length - 1) // block_length
    return max(256, num_blocks * 200)


def _log_sample_progress(
    sample_idx: int,
    batch_size: int,
    prompt_len: int,
    phase: str,
    elapsed: float | None = None,
    nfe: int | None = None,
) -> None:
    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    msg = (
        f"[RANK{rank}] multiblock sample {sample_idx + 1}/{batch_size} "
        f"prompt_tokens={prompt_len} {phase}"
    )
    if elapsed is not None:
        msg += f" elapsed={elapsed:.2f}s"
    if nfe is not None:
        msg += f" nfe={nfe}"
    print(msg, flush=True)
    log_dir = __import__("os").environ.get("DREAM_ROLLOUT_LOG_DIR", "").strip()
    if log_dir:
        import os

        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"rank{rank}.rollout.log"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _run_multiblock(
    module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gen_length: int,
    mb_cfg: DreamMultiBlockConfig,
    block_length: int,
    temperature: float,
    top_p: float,
    max_nfe: int,
) -> Tuple[torch.Tensor, int]:
    out, nfe = module.diffusion_generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen_length,
        temperature=temperature,
        top_p=top_p,
        threshold=mb_cfg.threshold,
        block_length=block_length,
        return_dict_in_generate=True,
        max_nfe=max_nfe,
    )
    if nfe >= max_nfe:
        try:
            rank = dist.get_rank()
        except Exception:
            rank = 0
        print(
            f"[RANK{rank}] WARN: multiblock hit max_nfe={max_nfe} (actual nfe={nfe}); "
            "generation may be incomplete",
            flush=True,
        )
    return _extract_sequences(out), nfe


def execute_dream_multiblock_generation(
    module,
    gen_kwargs: dict,
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
    process_outputs_fn,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """Run d3LLM multi-block rollout and delegate output parsing to ``process_outputs_fn``.

    d3LLM ``_sample_multi_block_kv_cache`` shares scalar block state across the batch
    (``handle_early_stop`` uses row-0 EOS for all rows; mask counts sum over batch).
    Official eval always uses batch_size=1. We therefore generate one compact sample
    at a time after stripping left padding.
    """
    gen_length = gen_kwargs["gen_length"]
    block_length = gen_kwargs["block_length"]
    temperature = gen_kwargs["temperature"]
    do_sample = gen_kwargs.get("do_sample", False)
    per_sample_seed = gen_kwargs.get("per_sample_seed", False)
    original_prompt_width = idx_repeat.size(1)
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

    use_fsdp_sync = False
    try:
        use_fsdp_sync = dist.is_initialized() and dist.get_world_size() > 1 and fsdp_version(module) > 0
    except Exception:
        pass
    set_fsdp_rollout_sync(use_fsdp_sync)

    device = idx_repeat.device
    compact_ids, compact_mask, prompt_lengths = _strip_left_padding(
        idx_repeat, attention_mask_repeat, pad_token_id
    )
    compact_ids = compact_ids.to(device)
    if compact_mask is not None:
        compact_mask = compact_mask.to(device)
    prompt_lengths = prompt_lengths.to(device)
    compact_target_len = int(prompt_lengths.max().item()) + response_length

    batch_size = idx_repeat.size(0)
    batch_start_time = time.time()
    top_p = 0.95 if (do_sample and temperature > 0) else (1.0 if temperature <= 0 else 0.95)
    total_nfe = 0
    max_nfe = gen_kwargs.get("max_nfe") or _estimate_max_nfe(gen_length, block_length)

    seq_list = []
    try:
        for i in range(batch_size):
            plen = int(prompt_lengths[i].item())
            t_sample = time.time()
            _log_sample_progress(i, batch_size, plen, "start")
            if per_sample_seed and do_sample and temperature > 0:
                torch.manual_seed(int(gen_kwargs.get("base_seed", 42)) + i)
            seq, nfe = _run_multiblock(
                module,
                compact_ids[i : i + 1],
                compact_mask[i : i + 1] if compact_mask is not None else None,
                gen_length,
                mb_cfg,
                block_length,
                temperature,
                top_p,
                max_nfe=max_nfe,
            )
            total_nfe += nfe
            _log_sample_progress(i, batch_size, plen, "done", time.time() - t_sample, nfe)
            seq_list.append(_pad_sequences_to_length(seq, compact_target_len, pad_token_id))
    finally:
        set_fsdp_rollout_sync(False)
    compact_outputs = torch.cat(seq_list, dim=0)

    outputs = _restore_padded_outputs(
        compact_outputs,
        prompt_lengths,
        original_prompt_width,
        response_length,
        pad_token_id,
    )

    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    elapsed = time.time() - batch_start_time
    avg_prompt = float(prompt_lengths.float().mean().item())
    print(
        f"[RANK{rank}] Dream multi-block generation for {batch_size} samples cost: "
        f"{elapsed:.2f}s (avg_prompt_tokens={avg_prompt:.0f}, total_nfe={total_nfe}, "
        f"bs=1 loop)",
        flush=True,
    )

    return process_outputs_fn(
        outputs, idx_repeat, attention_mask_repeat, response_length, tokenizer, module.device
    )
