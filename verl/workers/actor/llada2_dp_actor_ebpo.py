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
LLaDA2.x EBPO actor.

EBPO keeps the same block-causal forward pass as the llada2 BGPO path, but it
stores ELBO contributions only on the sampled response block and optimizes a
true block-level PPO objective.
"""

import logging
import os

import torch
import torch.nn.functional as F

from verl import DataProto
from verl.trainer.ppo.dllm_core_algos import agg_loss, compute_policy_loss_ebpo, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.llada2_dp_actor_bgpo import DLLMDataParallelPPOActor as BGPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DLLMDataParallelPPOActor(BGPOActor):
    def _forward_micro_batch(self, micro_batch, temperature, n_l, mc_num, calculate_entropy=False, call_fn_name=""):
        batch_size, seq_length = micro_batch["input_ids"].size(0), micro_batch["input_ids"].size(-1)
        response_length = micro_batch["responses"].size(-1)
        prompt_section_length = seq_length - response_length
        device = micro_batch["input_ids"].device

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = micro_batch["perturbed_seq"]
            mask_indices = micro_batch["mask_indices"]
            p_mask = micro_batch["p_mask"]
            attention_mask = micro_batch["attention_mask"]

            loss_per_token = torch.zeros((batch_size, mc_num, response_length), device=device)
            for i in range(mc_num):
                cur_perturbed_seq = perturbed_seq[:, i, :]
                cur_mask_indices = mask_indices[:, i, :]
                cur_p_mask = p_mask[:, i, :]

                compact_seq, compact_valid_mask, compact_target_mask, compact_p_mask, position_ids = self._compact_batch(
                    seq=cur_perturbed_seq,
                    attention_mask=attention_mask,
                    cur_mask_indices=cur_mask_indices,
                    cur_p_mask=cur_p_mask,
                )
                block_attention_mask = self._build_block_attention_mask(compact_valid_mask)

                logits = self.actor_module(
                    input_ids=compact_seq,
                    attention_mask=block_attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                ).logits

                for b in range(batch_size):
                    cur_len = int(compact_valid_mask[b].sum().item())
                    valid_logits = logits[b, :cur_len]
                    valid_targets = compact_seq[b, :cur_len]
                    valid_target_mask = compact_target_mask[b, :cur_len]
                    valid_p_mask = compact_p_mask[b, :cur_len]

                    if not valid_target_mask.any():
                        continue

                    compact_prompt_len = int(attention_mask[b, :prompt_section_length].sum().item())
                    compact_response_len = int(attention_mask[b, prompt_section_length:].sum().item())
                    if compact_response_len <= 0:
                        continue

                    response_start = compact_prompt_len
                    response_end = compact_prompt_len + compact_response_len
                    response_target_mask = valid_target_mask[response_start:response_end]
                    if not response_target_mask.any():
                        continue

                    active_block_local_idx = int(torch.nonzero(response_target_mask, as_tuple=False)[0].item())
                    active_block_id = active_block_local_idx // self.block_length
                    block_start = active_block_id * self.block_length
                    block_end = min(block_start + self.block_length, compact_response_len)

                    block_target_mask = response_target_mask[block_start:block_end]
                    if not block_target_mask.any():
                        continue

                    response_logits = valid_logits[response_start:response_end]
                    response_targets = valid_targets[response_start:response_end]
                    response_p_mask = valid_p_mask[response_start:response_end]

                    block_logits = response_logits[block_start:block_end]
                    block_targets = response_targets[block_start:block_end]
                    block_p_mask = response_p_mask[block_start:block_end]

                    block_token_losses = -(
                        F.cross_entropy(
                            block_logits[block_target_mask],
                            block_targets[block_target_mask],
                            reduction="none",
                        )
                        / block_p_mask[block_target_mask]
                    )
                    block_positions = torch.nonzero(block_target_mask, as_tuple=False).flatten() + block_start
                    loss_per_token[b, i, block_positions] = block_token_losses

            log_prob = loss_per_token.mean(dim=1)

        entropy = None
        if calculate_entropy:
            entropy = torch.zeros_like(log_prob)
            nonzero = log_prob != 0
            entropy[nonzero] = -log_prob[nonzero].exp() * log_prob[nonzero]

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
                    advantages = data["advantages"]

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
                        cur_data = {
                            **data,
                            "perturbed_seq": perturbed_seq[:, i : i + 1],
                            "mask_indices": mask_indices[:, i : i + 1],
                            "p_mask": p_mask[:, i : i + 1],
                        }
                        entropy, log_prob, loss_per_example = self._forward_micro_batch(
                            micro_batch=cur_data,
                            temperature=temperature,
                            n_l=1,
                            mc_num=1,
                            calculate_entropy=calculate_entropy,
                            call_fn_name="update_policy",
                        )

                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss_ebpo(
                            old_l_theta=old_loss_per_sample[:, i, :],
                            l_theta=loss_per_example[:, 0, :],
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_probs = cur_data["ref_log_probs"]
                            kld = kl_penalty(
                                l_theta=log_prob,
                                ref_l_theta=ref_log_probs,
                                kl_penalty=self.config.kl_loss_type,
                                advantages=advantages,
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
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
