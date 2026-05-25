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
"""Patch SGLang dLLM LowConfidence to match LMDeploy SDAR decoding.

LMDeploy uses ``low_confidence_dynamic`` with temperature sampling (see
``opencompass/opencompass/models/sdar_generate.py``).  SGLang's stock
``LowConfidence`` uses argmax and ignores ``temperature``, producing
identical rollouts for the same prompt and zero GRPO advantages.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_PATCH_APPLIED = False
_PATCH_DEFAULTS = {
    "confidence_threshold": 0.9,
    "denoising_steps": 4,
}


@dataclass
class SDARSamplingContext:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


_sdar_sampling_context: contextvars.ContextVar[Optional[SDARSamplingContext]] = contextvars.ContextVar(
    "sdar_sampling_context", default=None
)


@contextmanager
def sdar_sampling_context(*, temperature: float, top_k: int, top_p: float):
    """Thread-local sampling params consumed by the patched dLLM algorithm."""
    token = _sdar_sampling_context.set(
        SDARSamplingContext(
            temperature=temperature,
            top_k=_normalize_top_k(top_k),
            top_p=top_p,
        )
    )
    try:
        yield
    finally:
        _sdar_sampling_context.reset(token)


def _normalize_top_k(top_k: int) -> int:
    if top_k is None or top_k < 0:
        return 0
    return int(top_k)


def _get_sampling_context() -> SDARSamplingContext:
    ctx = _sdar_sampling_context.get()
    if ctx is None:
        return SDARSamplingContext()
    return ctx


def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool),
        -1,
        sorted_indices,
        sorted_mask,
    )
    return logits.masked_fill(mask_indices, float("-inf"))


def sample_with_temperature_topk_topp(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match ``sdar_generate.sample_with_temperature_topk_topp``."""
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)

    if temperature != 1.0:
        logits = logits / temperature

    original_probs = F.softmax(logits, dim=-1)

    filtered_logits = logits
    if top_k > 0:
        filtered_logits = _top_k_logits(filtered_logits, top_k)
    if top_p < 1.0:
        filtered_logits = _top_p_logits(filtered_logits, top_p)
    filtered_probs = F.softmax(filtered_logits, dim=-1)

    token = torch.multinomial(filtered_probs, num_samples=1)
    token_prob = torch.gather(original_probs, -1, token)
    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length: int, steps: int) -> torch.Tensor:
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


def apply_sdar_dllm_lmdeploy_patch(
    *,
    confidence_threshold: float = 0.9,
    denoising_steps: int = 4,
) -> None:
    """Monkey-patch SGLang ``LowConfidence`` once per process."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    _PATCH_DEFAULTS["confidence_threshold"] = confidence_threshold
    _PATCH_DEFAULTS["denoising_steps"] = denoising_steps

    from sglang.srt.dllm.algorithm import low_confidence as low_confidence_mod

    LowConfidence = low_confidence_mod.LowConfidence
    original_init = LowConfidence.__init__

    def patched_init(self, config):
        original_init(self, config)
        algo_cfg = config.algorithm_config or {}
        self.threshold = algo_cfg.get("threshold", _PATCH_DEFAULTS["confidence_threshold"])
        self.denoising_steps = int(algo_cfg.get("denoising_steps", _PATCH_DEFAULTS["denoising_steps"]))

    def patched_run(self, model_runner, forward_batch):
        batch_size = forward_batch.batch_size
        mask_index = forward_batch.input_ids == self.mask_id

        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            return logits_output, [], can_run_cuda_graph

        start_list = []
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        sampling = _get_sampling_context()
        num_transfer_tokens = get_num_transfer_tokens(self.block_size, self.denoising_steps)

        for step in range(self.denoising_steps + 1):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[curr_block_start:curr_block_end]
                block_mask_index = block_input_ids == self.mask_id
                if not block_mask_index.any():
                    continue

                curr_logits = logits_output.full_logits[curr_block_start:curr_block_end]
                x0, x0_p = sample_with_temperature_topk_topp(
                    curr_logits,
                    temperature=sampling.temperature,
                    top_k=sampling.top_k,
                    top_p=sampling.top_p,
                )

                confidence = torch.where(block_mask_index, x0_p, torch.tensor(-np.inf, device=x0_p.device))
                transfer_index = torch.zeros_like(block_mask_index, dtype=torch.bool)

                if step < len(num_transfer_tokens):
                    num_to_transfer = int(num_transfer_tokens[step].item())
                else:
                    num_to_transfer = 1

                high_conf_mask = confidence > self.threshold
                num_high_confidence = int(high_conf_mask.sum().item())
                if num_high_confidence >= num_to_transfer:
                    transfer_index = high_conf_mask
                else:
                    _, idx = torch.topk(confidence, num_to_transfer)
                    transfer_index[idx] = True

                x = torch.where(block_mask_index, x0, block_input_ids)
                block_input_ids[transfer_index] = x[transfer_index]

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [next_token_ids[i, start_list[i] :] for i in range(batch_size)]
        return logits_output, next_token_ids_list, can_run_cuda_graph

    LowConfidence.__init__ = patched_init
    LowConfidence.run = patched_run
    _PATCH_APPLIED = True
