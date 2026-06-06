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
"""Ensure Dream FullAttnMultiBlock honors rollout temperature (GRPO diversity)."""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def apply_dream_full_attn_multi_block_patch() -> None:
    """No-op when vendored SGLang already includes temperature sampling."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    from sglang.srt.dllm.algorithm import full_attn_multi_block as fam_mod

    source = inspect.getsource(fam_mod._sample_mask_tokens)
    if "multinomial" not in source or "decodable_vocab_size" not in source:
        raise RuntimeError(
            "FullAttnMultiBlock is missing Dream-safe temperature sampling; update "
            "third_party/sglang full_attn_multi_block.py (temperature + decodable_vocab_size)."
        )
    if "is_greedy" not in source:
        raise RuntimeError(
            "FullAttnMultiBlock must detect SGLang greedy (top_k<=1), not temperature<=0; "
            "update third_party/sglang full_attn_multi_block.py."
        )
    if "_mask_undecodable_logits" not in source:
        raise RuntimeError(
            "FullAttnMultiBlock must not clip logits on greedy/val argmax; update "
            "third_party/sglang full_attn_multi_block.py (_mask_undecodable_logits)."
        )
    fam_source = inspect.getsource(fam_mod)
    if "handle_early_stop" not in fam_source:
        raise RuntimeError(
            "FullAttnMultiBlock must include d3LLM handle_early_stop; update "
            "third_party/sglang full_attn_multi_block.py."
        )

    _PATCH_APPLIED = True
    logger.info("Dream FullAttnMultiBlock temperature sampling is active")
