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
LLaDA2.x BGPO actor.

Unlike the original packed LLaDA actor path, LLaDA2 should follow the same
block-diffusion training semantics validated in SFT:
1. compact each sample to its valid tokens,
2. build `[noisy_x, clean_x]`,
3. apply the official block-diffusion 4D mask,
4. score masked positions on the noisy half against clean-token targets.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from verl.workers.actor.llada_dp_actor_bgpo import DLLMDataParallelPPOActor as BaseDataParallelPPOActor


class DLLMDataParallelPPOActor(BaseDataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config, actor_module, actor_optimizer)
        self.block_length = config.get("block_length", 32)

    def _compact_batch(
        self,
        noisy_seq: torch.Tensor,
        clean_seq: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cur_mask_indices: torch.Tensor,
        cur_p_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = noisy_seq.size(0)
        lengths = attention_mask.sum(dim=1, dtype=torch.long)
        max_len = int(lengths.max().item())
        device = noisy_seq.device

        compact_noisy_seq = torch.full((batch_size, max_len), self.PAD_TOKEN_ID, dtype=noisy_seq.dtype, device=device)
        compact_clean_seq = torch.full((batch_size, max_len), self.PAD_TOKEN_ID, dtype=clean_seq.dtype, device=device)
        compact_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        compact_target_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        compact_p_mask = torch.zeros((batch_size, max_len), dtype=cur_p_mask.dtype, device=device)
        compact_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for b in range(batch_size):
            valid = attention_mask[b].bool()
            cur_len = int(lengths[b].item())
            compact_noisy_seq[b, :cur_len] = noisy_seq[b][valid]
            compact_clean_seq[b, :cur_len] = clean_seq[b][valid]
            compact_mask[b, :cur_len] = True
            compact_target_mask[b, :cur_len] = cur_mask_indices[b][valid]
            compact_p_mask[b, :cur_len] = cur_p_mask[b][valid]
            compact_position_ids[b, :cur_len] = position_ids[b][valid]

        return compact_noisy_seq, compact_clean_seq, compact_mask, compact_target_mask, compact_p_mask, compact_position_ids

    def _build_block_attention_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_len = valid_mask.shape
        device = valid_mask.device
        dtype = torch.float32

        full_len = max_len * 2
        q_idx = torch.arange(full_len, device=device)[:, None]
        kv_idx = torch.arange(full_len, device=device)[None, :]
        noisy_q = q_idx < max_len
        noisy_k = kv_idx < max_len
        block_q = torch.where(noisy_q, q_idx // self.block_length, (q_idx - max_len) // self.block_length)
        block_k = torch.where(noisy_k, kv_idx // self.block_length, (kv_idx - max_len) // self.block_length)

        block_diagonal = (block_q == block_k) & (noisy_q == noisy_k)
        offset_block_causal = (block_q > block_k) & (~noisy_k) & noisy_q
        block_causal = (block_q >= block_k) & (~noisy_k) & (~noisy_q)
        base_visible = block_diagonal | offset_block_causal | block_causal

        full_valid_mask = torch.cat((valid_mask, valid_mask), dim=1)
        valid_query = full_valid_mask[:, None, :, None]
        valid_key = full_valid_mask[:, None, None, :]
        visible = valid_query & valid_key & base_visible.unsqueeze(0).unsqueeze(0)

        attention_mask = torch.zeros((batch_size, 1, full_len, full_len), dtype=dtype, device=device)
        attention_mask.masked_fill_(~visible, torch.finfo(dtype).min)
        return attention_mask

    def _forward_micro_batch(self, micro_batch, temperature, n_l, mc_num, calculate_entropy=False, call_fn_name=""):
        batch_size, seq_length = micro_batch["input_ids"].size(0), micro_batch["input_ids"].size(-1)
        response_length = micro_batch["responses"].size(-1)
        device = micro_batch["input_ids"].device

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = micro_batch["perturbed_seq"]
            mask_indices = micro_batch["mask_indices"]
            p_mask = micro_batch["p_mask"]
            seq = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]

            loss_per_sample = torch.zeros((batch_size, mc_num), device=device)
            for i in range(mc_num):
                cur_perturbed_seq = perturbed_seq[:, i, :]
                cur_mask_indices = mask_indices[:, i, :]
                cur_p_mask = p_mask[:, i, :]

                compact_noisy_seq, compact_clean_seq, compact_valid_mask, compact_target_mask, compact_p_mask, compact_position_ids = self._compact_batch(
                    noisy_seq=cur_perturbed_seq,
                    clean_seq=seq,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cur_mask_indices=cur_mask_indices,
                    cur_p_mask=cur_p_mask,
                )
                block_attention_mask = self._build_block_attention_mask(compact_valid_mask)
                full_input_ids = torch.cat((compact_noisy_seq, compact_clean_seq), dim=1)
                full_position_ids = torch.cat((compact_position_ids, compact_position_ids), dim=1)

                logits = self.actor_module(
                    input_ids=full_input_ids,
                    attention_mask=block_attention_mask,
                    position_ids=full_position_ids,
                    return_dict=True,
                ).logits[:, : compact_noisy_seq.size(1), :]

                for b in range(batch_size):
                    cur_len = int(compact_valid_mask[b].sum().item())
                    valid_logits = logits[b, :cur_len]
                    valid_targets = compact_clean_seq[b, :cur_len]
                    valid_target_mask = compact_target_mask[b, :cur_len]
                    valid_p_mask = compact_p_mask[b, :cur_len]

                    if valid_target_mask.any():
                        loss_per_sample[b, i] = -(
                            F.cross_entropy(valid_logits[valid_target_mask], valid_targets[valid_target_mask], reduction="none")
                            / valid_p_mask[valid_target_mask]
                        ).sum()

            log_likelihood = loss_per_sample.sum(dim=1) / mc_num
            log_prob = log_likelihood.unsqueeze(-1).expand(-1, response_length)
            loss_per_sample = (loss_per_sample / response_length).unsqueeze(-1).expand(-1, -1, response_length).contiguous()

        entropy = None
        if calculate_entropy:
            prob = log_prob.exp()
            entropy = -prob * log_prob

        return entropy, log_prob, loss_per_sample
