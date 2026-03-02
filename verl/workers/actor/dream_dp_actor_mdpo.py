# Copyright 2025 Shanghai AI Lab
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
MDPO Actor for Dream model.
Extends the LLaDA MDPO actor with Dream-specific logit shifting (shift by 1 position).
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.mdpo_algos import compute_mdpo_policy_loss
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor import DataParallelPPOActor
from verl.workers.actor.llada_dp_actor_mdpo import DLLMDataParallelPPOActor as BaseDataParallelPPOActor
import torch.nn.functional as F

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor", "BaseDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(BaseDataParallelPPOActor):
    def _get_logits(self, model, packed_input, cu_seqlens, max_seqlen, prompt_len, cfg_scale=0.0, MASK_TOKEN_ID=126336):
        """
        Dream-specific logit computation with position shifting.
        packed_input: (1, total_seqlen)
        cu_seqlens: (batch_size+1,)
        max_seqlen: int
        prompt_len: (batch_size,) True prompt length of each sample
        """
        if cfg_scale > 0.:
            un_packed_input = packed_input.clone()
            for i in range(len(cu_seqlens) - 1):
                start = cu_seqlens[i].item()
                un_packed_input[0, start:start + prompt_len[i].item()] = MASK_TOKEN_ID
            packed_input_cat = torch.cat([packed_input, un_packed_input], dim=0)
            cu_seqlens_cat = torch.cat([cu_seqlens, cu_seqlens[1:] + cu_seqlens[-1]], dim=0)
            logits = model(packed_input_cat, cu_seqlens=cu_seqlens_cat, max_seqlen=max_seqlen).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(packed_input, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).logits
        logits = logits[:, :packed_input.shape[1]]
        # Dream-specific: shift logits by 1 position for causal LM loss computation
        shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1).contiguous()
        return shift_logits
