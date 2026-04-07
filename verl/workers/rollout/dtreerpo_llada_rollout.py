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
d-TreeRPO rollout for LLaDA that performs tree search with branching at
predefined diffusion steps. Builds a tree of partial generations and returns
all node data for reward computation and training segment construction.
"""

import uuid as uuid_mod

import torch
import torch.distributed as dist
from torch import nn
from tensordict import TensorDict

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.generate import add_gumbel_noise, get_num_transfer_tokens


__all__ = ["DTreeRPORollout"]


class DTreeRPORollout(BaseRollout):
    def __init__(self, module: nn.Module, config, tokenizer):
        """d-TreeRPO rollout for LLaDA that performs tree search generation."""
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = tokenizer
        self.MASK_TOKEN_ID = self.module.config.mask_token_id

        # diffusion related parameters
        self.response_length = config["response_length"]
        self.num_diffusion_steps = config["num_diffusion_steps"]
        self.block_length = config["block_length"]
        self.cfg_scale = config.get("cfg_scale", 0.0)
        self.temperature = config["temperature"]

        # model name for dream compatibility
        self.model_name = config.get("model_name", "llada")

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Perform d-TreeRPO tree search: build a tree of partial generations with
        branching at predefined diffusion steps.

        Returns DataProto containing tree structure for reward computation and
        training segment construction.
        """
        batch = prompts.batch
        input_ids = batch["input_ids"]  # (batch_size, prompt_len)
        attention_mask = batch["attention_mask"]
        batch_size = input_ids.size(0)
        prompt_length = input_ids.size(1)
        device = input_ids.device

        # Tree search config from meta_info (algorithm-specific params)
        response_length = prompts.meta_info.get("response_length", self.response_length)
        num_diffusion_steps = prompts.meta_info.get("num_diffusion_steps", self.num_diffusion_steps)
        block_length = prompts.meta_info.get("block_length", self.block_length)
        tree_branch_factor = prompts.meta_info.get("tree_branch_factor", 4)
        tree_contraction_factor = prompts.meta_info.get("tree_contraction_factor", 2)
        num_tree_samples = prompts.meta_info.get("num_tree_samples", 4)
        temperature = prompts.meta_info.get("temperature", self.temperature)
        cfg_scale = prompts.meta_info.get("cfg_scale", self.cfg_scale)
        remasking = prompts.meta_info.get("remasking", "low_confidence")
        MASK_TOKEN_ID = self.MASK_TOKEN_ID

        total_steps = num_diffusion_steps
        s = tree_contraction_factor
        branch_points = [total_steps] + [total_steps - int(k * total_steps / s) for k in range(1, s + 1)]
        branch_points = sorted(list(set(branch_points)), reverse=True)

        # Initialize root: prompt + all-mask completion
        seq_len = prompt_length + response_length
        initial_gen = torch.full((batch_size, seq_len), MASK_TOKEN_ID, dtype=torch.long, device=device)
        initial_gen[:, :prompt_length] = input_ids
        # Build attention mask for full sequence
        full_attn = torch.cat([
            attention_mask,
            torch.ones(batch_size, response_length, device=device, dtype=attention_mask.dtype)
        ], dim=1)

        prompt_index = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        prompt_index[:, :prompt_length] = True

        # Build tree
        root_id = "root"
        all_nodes = {
            root_id: {
                "id": root_id, "parent_id": None, "generation": initial_gen, "step": total_steps,
                "children": [], "value_vec": torch.zeros(batch_size, device=device), "is_leaf": False,
            }
        }
        current_level = [all_nodes[root_id]]

        self.module.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for i in range(len(branch_points) - 1):
                start_step, end_step = branch_points[i], branch_points[i + 1]
                steps_to_run = start_step - end_step
                next_level = []

                for node in current_level:
                    bf = num_tree_samples if node["id"] == root_id else tree_branch_factor
                    for k in range(bf):
                        new_gen = self._tree_generate_partial(
                            model=self.module,
                            start_generation=node["generation"],
                            prompt_index=prompt_index,
                            prompt_length=prompt_length,
                            response_length=response_length,
                            block_length=block_length,
                            current_step=start_step,
                            steps_to_run=steps_to_run,
                            total_diffusion_steps=total_steps,
                            temperature=temperature,
                            cfg_scale=cfg_scale,
                            remasking=remasking,
                            MASK_TOKEN_ID=MASK_TOKEN_ID,
                            full_attn=full_attn,
                        )
                        child_id = str(uuid_mod.uuid4())
                        child_node = {
                            "id": child_id, "parent_id": node["id"], "generation": new_gen,
                            "step": end_step, "children": [], "value_vec": None,
                            "is_leaf": (end_step == 0),
                        }
                        all_nodes[child_id] = child_node
                        node["children"].append(child_id)
                        next_level.append(child_node)

                current_level = next_level

        # Collect leaf generations for reward computation
        leaf_nodes = [n for n in all_nodes.values() if n["is_leaf"]]
        leaf_ids_list = []
        for leaf in leaf_nodes:
            leaf_ids_list.append(leaf["generation"])  # (batch_size, seq_len)

        if not leaf_ids_list:
            return DataProto.from_dict(
                tensors={"leaf_input_ids": torch.zeros(0, seq_len, device="cpu", dtype=torch.long)}
            )

        # Stack leaf_input_ids: (num_leaves, batch_size, seq_len)
        leaf_input_ids = torch.stack(leaf_ids_list, dim=0)
        leaf_responses = leaf_input_ids[:, :, prompt_length:]  # (num_leaves, batch_size, response_length)
        leaf_attn = full_attn.unsqueeze(0).expand(len(leaf_nodes), -1, -1)

        # Serialize tree structure
        node_ids = list(all_nodes.keys())
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        num_nodes = len(node_ids)

        node_generations = torch.stack([all_nodes[nid]["generation"] for nid in node_ids], dim=0)
        node_steps = torch.tensor([all_nodes[nid]["step"] for nid in node_ids], dtype=torch.long, device=device)
        node_is_leaf = torch.tensor([all_nodes[nid]["is_leaf"] for nid in node_ids], dtype=torch.bool, device=device)
        node_parent_idx = torch.tensor(
            [node_id_to_idx.get(all_nodes[nid]["parent_id"], -1) if all_nodes[nid]["parent_id"] else -1 for nid in node_ids],
            dtype=torch.long, device=device
        )
        max_children = max(len(all_nodes[nid]["children"]) for nid in node_ids)
        max_children = max(max_children, 1)
        node_children_idx = torch.full((num_nodes, max_children), -1, dtype=torch.long, device=device)
        for i, nid in enumerate(node_ids):
            for j, cid in enumerate(all_nodes[nid]["children"]):
                node_children_idx[i, j] = node_id_to_idx[cid]

        num_leaves_val = len(leaf_nodes)

        # DP_COMPUTE_PROTO splits/concats on dim0, which must be batch_size.
        # Reshape so dim0 = batch_size (per GPU), putting node/leaf dims into dim1+.
        node_gen_bt = node_generations.permute(1, 0, 2).contiguous().to("cpu")
        leaf_ids_bt = leaf_input_ids.permute(1, 0, 2).contiguous().to("cpu")
        leaf_resp_bt = leaf_responses.permute(1, 0, 2).contiguous().to("cpu")
        leaf_attn_bt = leaf_attn.permute(1, 0, 2).contiguous().to("cpu")

        # Tree structure: expand to (B, ...) for uniform batch dim
        node_steps_bt = node_steps.unsqueeze(0).expand(batch_size, -1).contiguous().to("cpu")
        node_is_leaf_bt = node_is_leaf.unsqueeze(0).expand(batch_size, -1).contiguous().to("cpu")
        node_parent_bt = node_parent_idx.unsqueeze(0).expand(batch_size, -1).contiguous().to("cpu")
        node_children_bt = node_children_idx.unsqueeze(0).expand(batch_size, -1, -1).contiguous().to("cpu")

        result = DataProto.from_dict(
            tensors={
                "node_generations": node_gen_bt,
                "node_steps": node_steps_bt,
                "node_is_leaf": node_is_leaf_bt,
                "node_parent_idx": node_parent_bt,
                "node_children_idx": node_children_bt,
                "leaf_input_ids": leaf_ids_bt,
                "leaf_responses": leaf_resp_bt,
                "leaf_attn": leaf_attn_bt,
            },
        )
        result.meta_info["batch_size_per_gpu"] = batch_size
        result.meta_info["prompt_length"] = prompt_length
        result.meta_info["num_leaves"] = num_leaves_val
        result.meta_info["num_nodes"] = num_nodes
        result.meta_info["branch_points"] = branch_points

        self.module.train()
        return result

    def _tree_generate_partial(
        self, model, start_generation, prompt_index, prompt_length, response_length,
        block_length, current_step, steps_to_run, total_diffusion_steps,
        temperature, cfg_scale, remasking, MASK_TOKEN_ID, full_attn,
    ):
        """Partially denoise from current_step for steps_to_run steps."""
        x = start_generation.clone()
        device = x.device
        gen_length = response_length
        num_blocks = gen_length // block_length
        steps_per_block = total_diffusion_steps // num_blocks

        target_step = max(0, current_step - steps_to_run)
        current_block_idx = (total_diffusion_steps - current_step) // steps_per_block
        target_block_idx = (total_diffusion_steps - target_step) // steps_per_block
        blocks_to_process = list(range(current_block_idx, min(target_block_idx + 1, num_blocks)))
        if not blocks_to_process:
            return x

        remaining_steps = steps_to_run

        for block_idx in blocks_to_process:
            if remaining_steps <= 0:
                break

            start_idx = prompt_length + block_idx * block_length
            end_idx = prompt_length + (block_idx + 1) * block_length

            block_step_start = total_diffusion_steps - block_idx * steps_per_block
            block_step_end = total_diffusion_steps - (block_idx + 1) * steps_per_block
            actual_start = min(current_step, block_step_start)
            actual_end = max(target_step, block_step_end)
            steps_needed = max(0, actual_start - actual_end)
            if steps_needed <= 0:
                continue

            done_in_block = max(0, min(block_step_start - current_step, steps_per_block))
            remain_in_block = steps_per_block - done_in_block
            if remain_in_block <= 0:
                continue

            steps_in_this_block = min(steps_needed, remaining_steps, remain_in_block)
            if steps_in_this_block <= 0:
                continue

            block_mask_now = x[:, start_idx:end_idx] == MASK_TOKEN_ID
            schedule = get_num_transfer_tokens(block_mask_now, remain_in_block)

            for step_i in range(steps_in_this_block):
                block_mask_step = x[:, start_idx:end_idx] == MASK_TOKEN_ID
                budget = schedule[:, step_i]
                mask_index_full = x == MASK_TOKEN_ID

                # Use packed input for efficient forward
                batch_size = x.size(0)
                packed = []
                cu_seqlens = [0]
                max_seqlen = 0
                for b in range(batch_size):
                    valid = x[b][full_attn[b] == 1]
                    packed.append(valid)
                    cu_seqlens.append(cu_seqlens[-1] + len(valid))
                    max_seqlen = max(max_seqlen, len(valid))
                packed_input = torch.cat(packed, dim=0).unsqueeze(0)
                cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

                if cfg_scale > 0.0:
                    un_packed = packed_input.clone()
                    for b in range(batch_size):
                        plen = full_attn[b, :prompt_length].sum().item()
                        un_packed[0, cu_seqlens[b]:cu_seqlens[b] + plen] = MASK_TOKEN_ID
                    packed_cat = torch.cat([packed_input, un_packed], dim=0)
                    cu_cat = torch.cat([cu_seqlens_t, cu_seqlens_t[1:] + cu_seqlens_t[-1]], dim=0)
                    logits = model(packed_cat, cu_seqlens=cu_cat, max_seqlen=max_seqlen).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(packed_input, cu_seqlens=cu_seqlens_t, max_seqlen=max_seqlen).logits

                # Unpack logits to padded form
                seq_len = x.size(1)
                full_logits = torch.zeros(batch_size, seq_len, logits.size(-1), device=device, dtype=logits.dtype)
                for b in range(batch_size):
                    full_logits[b, full_attn[b] == 1] = logits[0, cu_seqlens[b]:cu_seqlens[b + 1]]

                # Dream model requires shifted logits for correct token prediction
                if self.model_name == 'dream':
                    full_logits = torch.cat([full_logits[:, :1], full_logits[:, :-1]], dim=1)

                logits_with_noise = add_gumbel_noise(full_logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = torch.softmax(full_logits.float(), dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand_like(x0, dtype=torch.float32)
                else:
                    raise NotImplementedError(f"Remasking '{remasking}' not implemented")

                x0_p[:, :start_idx] = -float('inf')
                x0_p[:, end_idx:] = -float('inf')
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -torch.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
                for b in range(batch_size):
                    k_plan = int(budget[b].item())
                    if k_plan <= 0:
                        continue
                    k_avail = int(block_mask_step[b].sum().item())
                    if k_avail <= 0:
                        continue
                    k = min(k_plan, k_avail)
                    _, candidate_idx = torch.topk(confidence[b], k=k)
                    transfer_index[b, candidate_idx] = True

                transfer_index = transfer_index & ~prompt_index
                x[transfer_index] = x0[transfer_index]

            remaining_steps -= steps_in_this_block

        return x
