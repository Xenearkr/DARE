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
SDAR EBPO actor.

EBPO uses the same external forward-process artifacts as SDAR BGPO, but the
policy objective is applied on block-level ELBO contributions. SDAR already has
an official block-diffusion training path in `modeling_sdar.py`, so unlike
LLaDA2 we reuse that path directly and only request unreduced token losses for
the sampled response block.
"""

import logging
import os
from typing import Tuple

import torch
import torch.nn.functional as F

from verl import DataProto
from verl.trainer.ppo.dllm_core_algos import agg_loss, compute_policy_loss_ebpo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.sdar_dp_actor_bgpo import DLLMDataParallelPPOActor as BGPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(BGPOActor):
    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config, actor_module, actor_optimizer)
        self.block_length = int(config.get("block_length", 4))

    def _get_num_blocks(self, response_length: int) -> int:
        return (response_length + self.block_length - 1) // self.block_length

    def _build_block_response_mask(self, response_mask: torch.Tensor) -> torch.Tensor:
        batch_size, response_length = response_mask.shape
        num_blocks = self._get_num_blocks(response_length)
        pad_len = num_blocks * self.block_length - response_length
        if pad_len > 0:
            response_mask = F.pad(response_mask, (0, pad_len), value=0)
        return response_mask.view(batch_size, num_blocks, self.block_length).any(dim=-1)

    def _build_active_response_masks(
        self,
        sampled_token_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        response_mask = response_mask.bool()
        sampled_token_mask = sampled_token_mask.bool() & response_mask
        active_block_mask = self._build_block_response_mask(sampled_token_mask)

        batch_size, response_length = response_mask.shape
        num_blocks = active_block_mask.size(1)
        padded_len = num_blocks * self.block_length
        active_response_mask = (
            active_block_mask.unsqueeze(-1)
            .expand(batch_size, num_blocks, self.block_length)
            .reshape(batch_size, padded_len)
        )
        active_response_mask = active_response_mask[:, :response_length] & response_mask
        return active_block_mask, active_response_mask

    def _aggregate_token_to_blocks(self, token: torch.Tensor, token_mask: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        batch_size, response_length = token.shape
        num_blocks = self._get_num_blocks(response_length)
        pad_len = num_blocks * self.block_length - response_length
        if pad_len > 0:
            token = F.pad(token, (0, pad_len), value=0)
            token_mask = F.pad(token_mask, (0, pad_len), value=0)

        token = token.view(batch_size, num_blocks, self.block_length)
        token_mask = token_mask.view(batch_size, num_blocks, self.block_length).float()
        block = (token * token_mask).sum(dim=-1)
        if reduction == "sum":
            return block
        if reduction == "mean":
            denom = token_mask.sum(dim=-1).clamp_min(1.0)
            return block / denom
        raise ValueError(f"Unsupported reduction: {reduction}")

    def _forward_micro_batch(self, micro_batch, temperature, n_l, mc_num, calculate_entropy=False, call_fn_name="") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_length = micro_batch["input_ids"].size(0), micro_batch["input_ids"].size(-1)
        response_length = micro_batch["responses"].size(-1)
        prompt_length = seq_length - response_length
        device = micro_batch["input_ids"].device

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            position_ids = micro_batch["position_ids"]
            seq = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            perturbed_seq = micro_batch["perturbed_seq"]
            mask_indices = micro_batch["mask_indices"]
            p_mask = micro_batch["p_mask"]
            mc_num = perturbed_seq.shape[1]
            prompt_lens = attention_mask[:, :prompt_length].sum(dim=1)

            loss_per_token = torch.zeros((batch_size, mc_num, response_length), device=device)
            for b in range(batch_size):
                response_mask = attention_mask[b, -response_length:].bool()
                for i in range(mc_num):
                    response_target_mask = mask_indices[b, i, -response_length:].bool() & response_mask
                    masked_token_count = int(response_target_mask.sum().item())
                    if masked_token_count == 0:
                        continue

                    loss_b_i = self._get_logits(
                        model=self.actor_module,
                        seq=seq[b:b+1, :],
                        attention_mask=attention_mask[b:b+1, :],
                        position_ids=position_ids[b:b+1, :],
                        prompt_len=prompt_lens[b],
                        perturbed_seq=perturbed_seq[b:b+1, i, :],
                        mask_indices=mask_indices[b:b+1, i, :],
                        p_mask=p_mask[b:b+1, i, :],
                        cfg_scale=0.0,
                        MASK_TOKEN_ID=self.MASK_TOKEN_ID,
                    )
                    # SDAR forward returns diffusion NLL normalized by response_length.
                    # EBPO only samples one response block, so restoring the block ELBO sum
                    # only requires undoing that normalization and scattering the same block
                    # contribution across the masked tokens in the sampled block.
                    block_log_likelihood = (-loss_b_i) * response_length
                    loss_per_token[b, i, response_target_mask] = (
                        block_log_likelihood.to(loss_per_token.dtype) / masked_token_count
                    )

            log_likelihood = loss_per_token.mean(dim=1).sum(dim=-1)
            log_prob = log_likelihood.unsqueeze(-1).expand(-1, response_length)

        entropy = None
        if calculate_entropy:
            entropy = -log_prob.exp() * log_prob

        return entropy, log_prob, loss_per_token

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "old_loss_per_sample",
            "advantages",
            "perturbed_seq",
            "mask_indices",
            "p_mask",
        ]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_probs")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for data in dataloader:
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())

                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_loss_per_sample = data["old_loss_per_sample"]
                    advantages = self._aggregate_token_to_blocks(data["advantages"], response_mask, reduction="mean")

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = entropy_coeff != 0

                    accumulated_pg_loss = 0.0
                    accumulated_pg_clipfrac = 0.0
                    accumulated_ppo_kl = 0.0
                    accumulated_pg_clipfrac_lower = 0.0

                    perturbed_seq = data["perturbed_seq"]
                    mask_indices = data["mask_indices"]
                    p_mask = data["p_mask"]
                    mc_num = perturbed_seq.shape[1]
                    for i in range(mc_num):
                        cur_mask_indices = mask_indices[:, i, :]
                        active_block_mask, active_response_mask = self._build_active_response_masks(
                            sampled_token_mask=cur_mask_indices[:, -response_length:],
                            response_mask=response_mask,
                        )
                        cur_data = {
                            **data,
                            "perturbed_seq": perturbed_seq[:, i : i + 1],
                            "mask_indices": cur_mask_indices.unsqueeze(1),
                            "p_mask": p_mask[:, i : i + 1],
                        }
                        entropy, log_prob, loss_per_sample = self._forward_micro_batch(
                            micro_batch=cur_data,
                            temperature=temperature,
                            n_l=1,
                            mc_num=1,
                            calculate_entropy=calculate_entropy,
                            call_fn_name="update_policy",
                        )
                        old_loss_per_block = self._aggregate_token_to_blocks(
                            old_loss_per_sample[:, i, :],
                            response_mask,
                            reduction="sum",
                        )
                        loss_per_block = self._aggregate_token_to_blocks(
                            loss_per_sample[:, 0, :],
                            response_mask,
                            reduction="sum",
                        )
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_ebpo(
                            old_l_theta=old_loss_per_block,
                            l_theta=loss_per_block,
                            advantages=advantages,
                            response_mask=active_block_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(
                                loss_mat=entropy,
                                loss_mask=active_response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_probs = cur_data["ref_log_probs"]
                            kld = kl_penalty(
                                l_theta=log_prob,
                                ref_l_theta=ref_log_probs,
                                kl_penalty=self.config.kl_loss_type,
                                advantages=data["advantages"],
                            )
                            kl_loss = agg_loss(
                                loss_mat=kld,
                                loss_mask=active_response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        if self.config.use_dynamic_bsz:
                            loss = policy_loss * (cur_data["input_ids"].size(0) / self.config.ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation
                        loss /= self.mc_num
                        loss.backward()

                        accumulated_pg_loss += pg_loss.detach().item()
                        accumulated_pg_clipfrac += pg_clipfrac.detach().item()
                        accumulated_ppo_kl += ppo_kl.detach().item()
                        accumulated_pg_clipfrac_lower += pg_clipfrac_lower.detach().item()

                    append_to_dict(
                        metrics,
                        {
                            "actor/pg_loss": accumulated_pg_loss / mc_num,
                            "actor/pg_clipfrac": accumulated_pg_clipfrac / mc_num,
                            "actor/ppo_kl": accumulated_ppo_kl / mc_num,
                            "actor/pg_clipfrac_lower": accumulated_pg_clipfrac_lower / mc_num,
                        },
                    )

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
