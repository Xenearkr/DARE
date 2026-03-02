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
MDPO (Masked Diffusion Policy Optimization) algorithm functions.
"""

import torch
import torch.nn.functional as F


def compute_mdpo_policy_loss(
    per_token_logps,
    old_per_token_logps,
    advantages,
    completion_mask,
    confidence,
    max_completion_length,
    epsilon=0.2,
    beta=0.0,
    ref_per_token_logps=None,
    loss_agg_mode="token-mean",
):
    """
    Compute the MDPO policy loss with PPO clipping, lambda_t scaling, and confidence weighting.

    Args:
        per_token_logps (Tensor): (batch_size, seq_len) Current policy log probs.
        old_per_token_logps (Tensor): (batch_size, seq_len) Old policy log probs from rollout.
        advantages (Tensor): (batch_size,) Step-wise advantages for this diffusion step.
        completion_mask (Tensor): (batch_size, seq_len) Mask for valid completion tokens (1=valid, 0=pad).
        confidence (Tensor): (batch_size, seq_len) Confidence scores from the diffusion generation.
        max_completion_length (int): Maximum completion length (for lambda_t scaling).
        epsilon (float): PPO clip range.
        beta (float): KL coefficient. 0 to disable ref model KL.
        ref_per_token_logps (Tensor, optional): (batch_size, seq_len) Reference policy log probs.
        loss_agg_mode (str): Loss aggregation mode.

    Returns:
        Tuple of (pg_loss, pg_clipfrac, ppo_kl)
    """
    # Lambda_t scaling: accounts for mask ratio at each step
    # More masked tokens => larger lambda_t => more weight on this step
    num_valid = completion_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    lambda_t = max_completion_length / num_valid  # (batch_size, 1)

    # PPO clipped ratio
    ratio = torch.exp(per_token_logps - old_per_token_logps)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    # Expand advantages to token level: (batch_size,) -> (batch_size, 1)
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)

    # Clipped surrogate loss
    loss1 = -advantages * ratio * lambda_t
    loss2 = -advantages * clipped_ratio * lambda_t
    per_token_loss = torch.maximum(loss1, loss2)

    # Clip fraction metric
    pg_clipfrac = (loss2 > loss1).float()
    pg_clipfrac = (pg_clipfrac * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    # KL divergence (optional)
    ppo_kl = torch.tensor(0.0, device=per_token_logps.device)
    if beta > 0 and ref_per_token_logps is not None:
        logr = ref_per_token_logps - per_token_logps
        per_token_kl = logr ** 2 / 2
        per_token_loss = per_token_loss + beta * per_token_kl
        ppo_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    # Confidence-weighted loss
    weighted_loss = per_token_loss * completion_mask * confidence
    pg_loss = weighted_loss.sum() / completion_mask.sum().clamp(min=1e-4)

    return pg_loss, pg_clipfrac, ppo_kl


def compute_step_wise_advantage(all_step_rewards, num_generations):
    """
    Compute step-wise advantages from per-step rewards following MDPO paper (adv-v3 + adv-v4).

    Args:
        all_step_rewards (Tensor): (batch_size, diffusion_steps) Rewards at each diffusion step.
        num_generations (int): Number of generations per prompt (for GRPO normalization).

    Returns:
        Tensor: (batch_size, diffusion_steps) Step-wise advantages, GRPO-normalized.
    """
    batch_size, steps = all_step_rewards.shape
    device = all_step_rewards.device

    # adv-v3: reward difference + 1
    reward_diffs = torch.cat([
        all_step_rewards[:, 0:1],
        all_step_rewards[:, 1:] - all_step_rewards[:, :-1]
    ], dim=-1) + 1.0

    # adv-v4: add cumulative future rewards
    if steps > 1:
        future_rewards = all_step_rewards[:, 1:]  # (batch_size, steps-1)
        # Cumulative future reward: average of remaining rewards from each step
        flipped = future_rewards.flip(-1)
        cumsum = flipped.cumsum(-1)
        denom = torch.arange(1, steps, device=device).float()
        avg_future = (cumsum / denom).flip(-1)  # (batch_size, steps-1)
        cumulative = torch.cat([avg_future, all_step_rewards[:, -1:]], dim=-1)  # (batch_size, steps)
    else:
        cumulative = all_step_rewards

    advantages = reward_diffs + cumulative

    # GRPO normalization across generations
    if num_generations > 1 and batch_size >= num_generations:
        num_groups = batch_size // num_generations
        grouped = advantages.view(num_groups, num_generations, steps)
        mean = grouped.mean(dim=1, keepdim=True)
        std = grouped.std(dim=1, keepdim=True)
        grouped = (grouped - mean) / (std + 1e-4)
        advantages = grouped.view(batch_size, steps)

    return advantages


def select_top_k_steps(advantages, k):
    """
    Select top-K diffusion steps with highest absolute advantage for training.

    Args:
        advantages (Tensor): (batch_size, diffusion_steps) Step-wise advantages.
        k (int): Number of steps to select.

    Returns:
        Tensor: (k,) Indices of selected steps (sorted).
    """
    steps = advantages.shape[1]
    k = min(k, steps)

    # Sum absolute advantages across batch to get step importance
    step_importance = advantages.abs().sum(dim=0)  # (diffusion_steps,)
    _, top_k_indices = torch.topk(step_importance, k=k)
    top_k_indices = top_k_indices.sort().values  # Sort for sequential processing

    return top_k_indices
