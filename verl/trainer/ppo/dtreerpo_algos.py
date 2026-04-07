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
d-TreeRPO (Diffusion Tree Reward Propagation Optimization) algorithm functions.

Core algorithm logic for computing rewards at leaf nodes, propagating values
through the tree, computing local advantages, and constructing training segments.
"""

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_reward


def compute_dtreerpo_rewards_and_segments(
    tree_output,
    batch,
    reward_fn,
    response_length_cfg,
    micro_batch_size,
    temperature,
):
    """
    Given tree search output, compute rewards at leaf nodes, propagate upward,
    compute local advantages, and construct training segments.

    Args:
        tree_output (DataProto): Output from tree search generation containing tree structure.
        batch (DataProto): Original batch with non_tensor_batch for reward computation.
        reward_fn: Reward function for computing leaf rewards.
        response_length_cfg (int): Configured response length for dummy fields.
        micro_batch_size (int): Micro batch size for log prob computation.
        temperature (float): Temperature for log prob computation.

    Returns:
        Tuple of (segments DataProto or None, metrics dict).
    """
    tree_batch = tree_output.batch
    meta = tree_output.meta_info
    prompt_length = meta["prompt_length"]
    num_leaves = meta["num_leaves"]
    num_nodes = meta["num_nodes"]
    branch_points = meta["branch_points"]

    # After DP concat, dim0 = total_batch_size.
    # Tensors: (total_batch, num_nodes/num_leaves, ...)
    # Tree structure is the same for all batch items; take from first item.
    batch_size = tree_batch["node_generations"].size(0)  # total batch size after DP gather

    node_generations = tree_batch["node_generations"]  # (B, num_nodes, seq_len)
    node_steps = tree_batch["node_steps"][0]           # (num_nodes,) - same for all batch items
    node_is_leaf = tree_batch["node_is_leaf"][0]       # (num_nodes,)
    node_parent_idx = tree_batch["node_parent_idx"][0] # (num_nodes,)
    node_children_idx = tree_batch["node_children_idx"][0]  # (num_nodes, max_children)

    leaf_input_ids_bt = tree_batch["leaf_input_ids"]   # (B, num_leaves, seq_len)
    leaf_responses_bt = tree_batch["leaf_responses"]   # (B, num_leaves, resp_len)
    leaf_attn_bt = tree_batch["leaf_attn"]             # (B, num_leaves, seq_len)

    # Transpose node_generations to (num_nodes, B, seq_len) for tree processing
    node_generations = node_generations.permute(1, 0, 2).contiguous()  # (num_nodes, B, seq_len)

    seq_len = node_generations.size(-1)
    response_length = seq_len - prompt_length
    device = node_generations.device

    # Flatten leaf data to (num_leaves * B, ...) for reward computation
    leaf_input_ids = leaf_input_ids_bt.permute(1, 0, 2).contiguous().reshape(-1, seq_len)  # (num_leaves * B, seq_len)
    leaf_responses = leaf_responses_bt.permute(1, 0, 2).contiguous().reshape(-1, response_length)
    leaf_attn = leaf_attn_bt.permute(1, 0, 2).contiguous().reshape(-1, seq_len)

    # Compute rewards for leaf nodes
    leaf_total = num_leaves * batch_size
    # Expand non_tensor_batch to match leaf batch size before passing to DataProto
    expanded_non_tensors = {}
    if batch.non_tensor_batch is not None:
        for key, val in batch.non_tensor_batch.items():
            if isinstance(val, np.ndarray):
                # Repeat each element num_leaves times (leaves are ordered leaf-major)
                expanded_non_tensors[key] = np.tile(val, num_leaves)[:leaf_total]
            else:
                expanded_non_tensors[key] = val

    leaf_batch = DataProto.from_dict(
        tensors={
            "input_ids": leaf_input_ids,
            "responses": leaf_responses,
            "prompts": leaf_input_ids[:, :prompt_length],
            "attention_mask": leaf_attn,
            "position_ids": leaf_attn.cumsum(dim=-1) - 1,
        },
        non_tensors=expanded_non_tensors,
    )

    reward_tensor, reward_extra_infos_dict = compute_reward(leaf_batch, reward_fn)
    leaf_rewards = reward_tensor.sum(dim=-1)  # (num_leaves * batch_size,)

    # Assign rewards to leaf nodes: reshape to (num_leaves, batch_size)
    leaf_rewards_mat = leaf_rewards.reshape(num_leaves, batch_size)

    # Create value vectors for all nodes
    node_value_vecs = torch.zeros(num_nodes, batch_size, device=device)

    # Assign leaf values
    leaf_idx = 0
    for i in range(num_nodes):
        if node_is_leaf[i]:
            node_value_vecs[i] = leaf_rewards_mat[leaf_idx]
            leaf_idx += 1

    # Bottom-up propagation: process steps from lowest to highest
    for step in sorted(branch_points[:-1]):
        for i in range(num_nodes):
            if node_steps[i] == step and i > 0:  # skip root (i=0)
                children = node_children_idx[i]
                valid_children = children[children >= 0]
                if len(valid_children) > 0:
                    child_vals = torch.stack([node_value_vecs[c] for c in valid_children])
                    node_value_vecs[i] = torch.nanmean(child_vals, dim=0)

    # Root propagation
    root_children = node_children_idx[0]
    valid_root_children = root_children[root_children >= 0]
    if len(valid_root_children) > 0:
        root_child_vals = torch.stack([node_value_vecs[c] for c in valid_root_children])
        node_value_vecs[0] = torch.nanmean(root_child_vals, dim=0)

    # Compute local advantages for each non-root node
    node_local_adv = torch.zeros(num_nodes, batch_size, device=device)
    for i in range(1, num_nodes):
        parent = node_parent_idx[i].item()
        siblings = node_children_idx[parent]
        valid_siblings = siblings[(siblings >= 0) & (siblings != i)]
        if len(valid_siblings) > 0:
            sib_vals = torch.stack([node_value_vecs[s] for s in valid_siblings])
            sib_mean = torch.nanmean(sib_vals, dim=0)
        else:
            sib_mean = torch.zeros(batch_size, device=device)
        node_local_adv[i] = node_value_vecs[i] - sib_mean

    # Build training segments
    all_parent_ids = []
    all_child_ids = []
    all_attention_masks = []
    all_local_advantages = []
    all_group_ids = []

    group_key_to_id = {}
    next_gid = 0

    for i in range(1, num_nodes):
        parent_idx_val = node_parent_idx[i].item()
        parent_gen = node_generations[parent_idx_val]   # (batch_size, seq_len)
        child_gen = node_generations[i]                 # (batch_size, seq_len)
        adv = node_local_adv[i]                         # (batch_size,)

        # Build attention mask
        attn = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

        for b in range(batch_size):
            key = (parent_idx_val, b)
            gid = group_key_to_id.get(key)
            if gid is None:
                gid = next_gid
                next_gid += 1
                group_key_to_id[key] = gid
            all_group_ids.append(gid)

        all_parent_ids.append(parent_gen)
        all_child_ids.append(child_gen)
        all_attention_masks.append(attn)
        all_local_advantages.append(adv)

    if not all_parent_ids:
        return None, {"dtreerpo/mean_leaf_reward": 0.0}

    parent_ids_cat = torch.cat(all_parent_ids, dim=0)
    child_ids_cat = torch.cat(all_child_ids, dim=0)
    attn_cat = torch.cat(all_attention_masks, dim=0)
    adv_cat = torch.cat(all_local_advantages, dim=0)
    group_ids_cat = torch.tensor(all_group_ids, dtype=torch.long, device=device)
    prompt_length_t = torch.tensor([prompt_length], dtype=torch.long, device=device).expand(parent_ids_cat.size(0))

    segments = DataProto.from_dict(
        tensors={
            "parent_ids": parent_ids_cat,
            "child_ids": child_ids_cat,
            "attention_mask": attn_cat,
            "local_advantages": adv_cat,
            "group_ids": group_ids_cat,
            "prompt_length": prompt_length_t,
            # Dummy fields for metrics compatibility
            "prompts": parent_ids_cat[:, :prompt_length],
            "responses": child_ids_cat[:, prompt_length:],
            "token_level_scores": torch.zeros(parent_ids_cat.size(0), response_length_cfg, device=device),
            "token_level_rewards": torch.zeros(parent_ids_cat.size(0), response_length_cfg, device=device),
            "response_mask": attn_cat[:, -response_length_cfg:],
        },
    )
    segments.meta_info["micro_batch_size"] = micro_batch_size
    segments.meta_info["temperature"] = temperature

    dtreerpo_metrics = {
        "dtreerpo/mean_leaf_reward": leaf_rewards.mean().item(),
        "dtreerpo/mean_root_value": node_value_vecs[0].mean().item(),
        "dtreerpo/mean_local_adv": adv_cat.mean().item(),
        "dtreerpo/num_segments": parent_ids_cat.size(0),
    }

    # Also set reward tensors on batch for standard metrics
    # Use leaf rewards averaged per prompt as the reward signal
    mean_leaf_reward_per_prompt = leaf_rewards_mat.mean(dim=0)  # (batch_size,)
    if reward_extra_infos_dict:
        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

    return segments, dtreerpo_metrics
