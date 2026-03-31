import numpy as np
from verl.trainer.ppo.ray_trainer import *
from verl.trainer.ppo.ray_trainer import _timer
from verl.trainer.ppo.dllm_metric_utils import (
    process_validation_metrics,
)
from verl.trainer.ppo.mdpo_algos import compute_step_wise_advantage, select_top_k_steps


class DLLMRayPPOTrainer(RayPPOTrainer):
    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def _dtreerpo_compute_rewards_and_segments(self, tree_output, batch, gen_batch):
        """
        Given tree search output, compute rewards at leaf nodes, propagate upward,
        compute local advantages, and construct training segments.
        """
        import torch

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
                    # Repeat each element num_leaves times (leaves are ordered leaf-major: all batch items for leaf0, then leaf1, etc.)
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

        reward_tensor, reward_extra_infos_dict = compute_reward(leaf_batch, self.reward_fn)
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

        response_length_cfg = self.config.actor_rollout_ref.rollout.get("response_length")
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

        from tensordict import TensorDict
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
        segments.meta_info["micro_batch_size"] = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
        segments.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

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

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
               
                gen_batch.non_tensor_batch["reward_model"] = batch.non_tensor_batch["reward_model"].copy()

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # d-TreeRPO skips standard rollout; tree search handles generation
                    if self.config.algorithm.name == "dtreerpo":
                        gen_batch_output = None
                    else:
                        # generate a batch
                        with _timer("gen", timing_raw):
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                self.async_rollout_manager.wake_up()
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                self.async_rollout_manager.sleep()

                    if self.config.algorithm.name != "dtreerpo":
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            with _timer("gen_max", timing_raw):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                batch.batch["reward_baselines"] = reward_baseline_tensor

                                del gen_baseline_batch, gen_baseline_output

                        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                        batch.batch["response_mask"] = compute_response_mask(batch)
                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        with _timer("reward", timing_raw):
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    if self.config.algorithm.name in ["d1", "coupled-grpo", "bgpo", "ebpo", "spg"]:
                        if self.config.actor_rollout_ref.model.name != "sdar":
                            with _timer("forward_process", timing_raw):
                                forward_batch_output = self.actor_rollout_wg.forward_process(batch)
                            batch = batch.union(forward_batch_output)
                        if self.config.algorithm.name in ["bgpo", "ebpo"]:
                            # recompute old_log_probs
                            with _timer("old_log_prob", timing_raw):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["old_entropys"]
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("old_entropys")
                                batch = batch.union(old_log_prob)

                                if self.config.algorithm.name == "bgpo" and "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                    actor_old_log_probs = batch.batch["old_log_probs"]
                                    attention_mask = batch.batch["attention_mask"]
                                    responses = batch.batch["responses"]
                                    response_length = responses.size(1)
                                    response_mask = attention_mask[:, -response_length:]

                                    rollout_probs = torch.exp(rollout_old_log_probs)
                                    actor_probs = torch.exp(actor_old_log_probs)
                                    rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                    rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                    rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                    rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                    rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                    metrics.update(
                                        {
                                            "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                            "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                            "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                        }
                                    )
                        elif self.config.algorithm.name in ["d1"]:
                            # recompute old_log_probs
                            with _timer("old_log_prob", timing_raw):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["old_entropys"]
                                response_masks = batch.batch["mask_indices"][:, :, -entropys.shape[-1]:]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("old_entropys")
                                batch = batch.union(old_log_prob)

                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                    actor_old_log_probs = batch.batch["old_log_probs"]
                                    attention_mask = batch.batch["attention_mask"]
                                    responses = batch.batch["responses"]
                                    response_length = responses.size(1)
                                    response_mask = attention_mask[:, -response_length:]

                                    rollout_probs = torch.exp(rollout_old_log_probs)
                                    actor_probs = torch.exp(actor_old_log_probs)
                                    rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                    rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                    rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                    rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                    rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                    metrics.update(
                                        {
                                            "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                            "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                            "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                        }
                                    )
                        elif self.config.algorithm.name in ["coupled-grpo"]:
                            # recompute old_log_probs
                            with _timer("old_log_prob", timing_raw):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["old_entropys"]
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_loss = agg_loss(loss_mat=entropys[:, -response_masks.shape[-1]:], loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("old_entropys")
                                batch = batch.union(old_log_prob)

                                if "rollout_log_probs" in batch.batch.keys():
                                    # TODO: we may want to add diff of probs too.
                                    rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                    actor_old_log_probs = batch.batch["old_log_probs"]
                                    attention_mask = batch.batch["attention_mask"]
                                    responses = batch.batch["responses"]
                                    response_length = responses.size(1)
                                    response_mask = attention_mask[:, -response_length:]

                                    rollout_probs = torch.exp(rollout_old_log_probs)
                                    actor_probs = torch.exp(actor_old_log_probs)
                                    rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                    rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                    rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                    rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                    rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                    metrics.update(
                                        {
                                            "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                            "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                            "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                        }
                                    )
                        else:
                            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    
                    elif self.config.algorithm.name == "cj-grpo":
                        # recompute old_log_probs
                        with _timer("old_log_prob", timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["old_entropys"]
                            response_masks = batch.batch["reversed_traj_unmask_positions"][:, :, -entropys.shape[-1]:]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("old_entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )

                    elif self.config.algorithm.name == "mdpo":
                        # MDPO: compute per-step rewards, step-wise advantages, select top-K steps
                        with _timer("mdpo_step_rewards", timing_raw):
                            all_steps_completion_ids = batch.batch["all_steps_completion_ids"]  # (batch_size, steps, response_len)
                            prompts_ids = batch.batch["prompts"]  # (batch_size, prompt_len)
                            mdpo_batch_size, num_diffusion_steps, response_len = all_steps_completion_ids.shape

                            # Compute reward at each diffusion step
                            all_step_rewards = []
                            for t in range(num_diffusion_steps):
                                t_completion_ids = all_steps_completion_ids[:, t, :]  # (batch_size, response_len)
                                t_input_ids = torch.cat([prompts_ids, t_completion_ids], dim=1)

                                # Create a temporary batch for reward computation
                                t_batch = DataProto.from_dict(
                                    tensors={
                                        "input_ids": t_input_ids,
                                        "responses": t_completion_ids,
                                        "prompts": prompts_ids,
                                        "attention_mask": batch.batch["attention_mask"],
                                        "position_ids": batch.batch["position_ids"],
                                    },
                                    non_tensors=batch.non_tensor_batch,
                                )
                                reward_t, _ = compute_reward(t_batch, self.reward_fn)
                                step_reward = reward_t.sum(dim=-1)  # (batch_size,)
                                all_step_rewards.append(step_reward)

                            all_step_rewards_tensor = torch.stack(all_step_rewards, dim=-1)  # (batch_size, steps)

                        with _timer("mdpo_advantages", timing_raw):
                            # Compute step-wise advantages (GRPO normalization via UID grouping)
                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            step_advantages = compute_step_wise_advantage(
                                all_step_rewards_tensor,
                                index=batch.non_tensor_batch["uid"],
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            )  # (batch_size, steps)

                            # Select top-K steps for training
                            sample_train_steps = self.config.actor_rollout_ref.actor.get("sample_train_steps", 16)
                            top_k_indices = select_top_k_steps(step_advantages, k=sample_train_steps)
                            print(f"MDPO: selected {len(top_k_indices)} steps for training: {top_k_indices.tolist()}")

                            # Also compute final-step reward for standard metrics
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor
                            batch.batch["token_level_rewards"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # Log MDPO-specific metrics
                            mdpo_metrics = {
                                "mdpo/mean_step_reward": all_step_rewards_tensor.mean().item(),
                                "mdpo/final_step_reward": all_step_rewards_tensor[:, -1].mean().item(),
                                "mdpo/mean_advantage": step_advantages.mean().item(),
                                "mdpo/num_train_steps": len(top_k_indices),
                            }
                            metrics.update(mdpo_metrics)

                        # For each selected step, pack data for actor update
                        with _timer("mdpo_pack_steps", timing_raw):
                            all_steps_input_ids = batch.batch["all_steps_input_ids"]  # (batch, steps, response_len)
                            all_confidence = batch.batch["all_confidence"]  # (batch, steps, response_len)
                            attention_mask = batch.batch["attention_mask"]  # (batch, seq_len)

                            # Build packed MDPO training data across all selected steps
                            mdpo_step_input_ids_list = []
                            mdpo_step_target_ids_list = []
                            mdpo_advantages_list = []
                            mdpo_confidence_list = []
                            mdpo_completion_mask_list = []
                            mdpo_attention_mask_list = []

                            for step_idx in top_k_indices:
                                step_idx = step_idx.item()
                                # Build full input_ids: prompt + corrupted completion at this step
                                step_input = torch.cat([prompts_ids, all_steps_input_ids[:, step_idx, :]], dim=1)
                                # Build full target_ids: prompt + denoised completion at this step
                                step_target = torch.cat([prompts_ids, all_steps_completion_ids[:, step_idx, :]], dim=1)
                                step_adv = step_advantages[:, step_idx]  # (batch_size,)
                                step_conf = all_confidence[:, step_idx, :]  # (batch_size, response_len)

                                # Completion mask: where the step input is masked (needs prediction)
                                mask_token_id = batch.meta_info["mask_token_id"]
                                completion_mask = (all_steps_input_ids[:, step_idx, :] == mask_token_id).float()

                                mdpo_step_input_ids_list.append(step_input)
                                mdpo_step_target_ids_list.append(step_target)
                                mdpo_advantages_list.append(step_adv)
                                mdpo_confidence_list.append(step_conf)
                                mdpo_completion_mask_list.append(completion_mask)
                                mdpo_attention_mask_list.append(attention_mask)

                            # Concatenate all steps into one big batch
                            mdpo_step_input_ids = torch.cat(mdpo_step_input_ids_list, dim=0)
                            mdpo_step_target_ids = torch.cat(mdpo_step_target_ids_list, dim=0)
                            mdpo_advantages = torch.cat(mdpo_advantages_list, dim=0)
                            mdpo_confidence = torch.cat(mdpo_confidence_list, dim=0)
                            mdpo_completion_mask = torch.cat(mdpo_completion_mask_list, dim=0)
                            mdpo_attention_mask = torch.cat(mdpo_attention_mask_list, dim=0)

                        # Compute old log probs for all packed steps
                        with _timer("mdpo_old_log_prob", timing_raw):
                            mdpo_data = DataProto.from_dict(
                                tensors={
                                    "mdpo_step_input_ids": mdpo_step_input_ids,
                                    "mdpo_step_target_ids": mdpo_step_target_ids,
                                    "attention_mask": mdpo_attention_mask,
                                    "completion_mask": mdpo_completion_mask,
                                },
                            )
                            mdpo_data.meta_info["micro_batch_size"] = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                            mdpo_data.meta_info["max_token_len"] = self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
                            mdpo_data.meta_info["use_dynamic_bsz"] = self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
                            mdpo_data.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                            old_log_prob_output = self.actor_rollout_wg.compute_log_prob(mdpo_data)
                            old_per_token_logps = old_log_prob_output.batch["old_log_probs"]

                        # Build the final batch for actor update by updating the existing batch
                        # This preserves the existing meta_info from line 228

                        # For MDPO, compute returns from final step rewards for metrics
                        # MDPO doesn't use standard returns, but compute_data_metrics expects this key
                        final_rewards = all_step_rewards_tensor[:, -1]  # (batch_size,)
                        # Expand final_rewards to match top-K repeated batch size
                        final_rewards_expanded = final_rewards.repeat(len(top_k_indices))  # (batch_size * k,)

                        # For MDPO, advantages are per-sample scalars.
                        # Expand to per-token advantages with dim=1 = response_length (not seq_len!)
                        # This matches standard algorithms where advantages shape is (batch_size, response_length)
                        response_length = mdpo_completion_mask.size(-1)  # Length of response portion only

                        # Expand advantages and returns to (batch_size, response_length)
                        # Both need to be 2D for compatibility with metrics and response_mask
                        mdpo_advantages_expanded = mdpo_advantages.unsqueeze(-1).expand(-1, response_length)
                        mdpo_returns_expanded = final_rewards_expanded.unsqueeze(-1).expand(-1, response_length)

                        # Create new tensor dict for MDPO
                        mdpo_tensors = {
                            "mdpo_step_input_ids": mdpo_step_input_ids,
                            "mdpo_step_target_ids": mdpo_step_target_ids,
                            "attention_mask": mdpo_attention_mask,
                            "completion_mask": mdpo_completion_mask,
                            "response_mask": mdpo_completion_mask,
                            "advantages": mdpo_advantages_expanded,  # (batch_size, response_length)
                            "returns": mdpo_returns_expanded,  # (batch_size, response_length)
                            "confidence": mdpo_confidence,
                            "old_per_token_logps": old_per_token_logps,
                            # Keep these for metrics computation
                            "prompts": prompts_ids.repeat(len(top_k_indices), 1),
                            "responses": all_steps_completion_ids[:, -1, :].repeat(len(top_k_indices), 1),
                            "token_level_scores": reward_tensor.repeat(len(top_k_indices), 1),
                            "token_level_rewards": reward_tensor.repeat(len(top_k_indices), 1),
                        }

                        # Replace the batch tensors (keeps existing meta_info!)
                        from tensordict import TensorDict
                        new_batch_size = mdpo_step_input_ids.shape[0]
                        batch.batch = TensorDict(source=mdpo_tensors, batch_size=(new_batch_size,), device=mdpo_step_input_ids.device)

                        # Expand non_tensor_batch to match the repeated batch size
                        if batch.non_tensor_batch is not None:
                            for key, val in batch.non_tensor_batch.items():
                                if isinstance(val, np.ndarray):
                                    # Repeat array along batch dimension to match new batch size
                                    batch.non_tensor_batch[key] = np.repeat(val, len(top_k_indices), axis=0)

                        # Update only the meta_info keys that are different for MDPO
                        batch.meta_info["global_token_num"] = torch.sum(mdpo_attention_mask, dim=-1).tolist()
                        batch.meta_info["micro_batch_size"] = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
                        batch.meta_info["max_token_len"] = self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
                        batch.meta_info["use_dynamic_bsz"] = self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                        # Skip standard forward_process, compute_log_prob, compute_advantage
                        # Go directly to actor update
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with _timer("update_actor", timing_raw):
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                    elif self.config.algorithm.name == "dtreerpo":
                        # d-TreeRPO: tree search, reward propagation, local advantage, actor update
                        with _timer("dtreerpo_tree_search", timing_raw):
                            tree_search_batch = DataProto.from_dict(
                                tensors={
                                    "input_ids": gen_batch.batch["input_ids"],
                                    "attention_mask": gen_batch.batch["attention_mask"],
                                },
                            )
                            tree_search_batch.meta_info.update({
                                "response_length": self.config.actor_rollout_ref.rollout.get("response_length"),
                                "num_diffusion_steps": self.config.actor_rollout_ref.rollout.get("num_diffusion_steps"),
                                "block_length": self.config.actor_rollout_ref.rollout.get("block_length"),
                                "tree_branch_factor": self.config.actor_rollout_ref.actor.get("tree_branch_factor", 4),
                                "tree_contraction_factor": self.config.actor_rollout_ref.actor.get("tree_contraction_factor", 2),
                                "num_tree_samples": self.config.actor_rollout_ref.actor.get("num_tree_samples", 4),
                                "temperature": self.config.actor_rollout_ref.rollout.temperature,
                                "cfg_scale": self.config.actor_rollout_ref.rollout.get("cfg_scale", 0.0),
                                "remasking": "low_confidence",
                            })
                            # Pad for DP
                            tree_search_batch_padded, tree_pad_size = pad_dataproto_to_divisor(
                                tree_search_batch, self.actor_rollout_wg.world_size
                            )
                            tree_output_padded = self.actor_rollout_wg.tree_search_generate(tree_search_batch_padded)
                            tree_output = unpad_dataproto(tree_output_padded, pad_size=tree_pad_size)

                        with _timer("dtreerpo_reward_propagation", timing_raw):
                            dtreerpo_segments, dtreerpo_reward_metrics = self._dtreerpo_compute_rewards_and_segments(
                                tree_output, batch, gen_batch
                            )
                            metrics.update(dtreerpo_reward_metrics)

                        if dtreerpo_segments is not None and len(dtreerpo_segments.batch) > 0:
                            # Compute old local log probs
                            with _timer("dtreerpo_old_log_prob", timing_raw):
                                old_logps_output = self.actor_rollout_wg.compute_dtreerpo_log_prob(dtreerpo_segments)
                                dtreerpo_segments.batch["old_local_logps"] = old_logps_output.batch["old_local_logps"]

                            # Compute ref log probs if KL is used
                            if self.config.actor_rollout_ref.actor.use_kl_loss:
                                with _timer("dtreerpo_ref_log_prob", timing_raw):
                                    dtreerpo_segments.meta_info["is_lora"] = True
                                    ref_logps_output = self.actor_rollout_wg.compute_dtreerpo_log_prob(dtreerpo_segments)
                                    dtreerpo_segments.batch["ref_local_logps"] = ref_logps_output.batch["old_local_logps"]
                                    dtreerpo_segments.meta_info.pop("is_lora", None)

                            # Actor update
                            if self.config.trainer.critic_warmup <= self.global_steps:
                                with _timer("update_actor", timing_raw):
                                    dtreerpo_segments.meta_info["multi_turn"] = False
                                    dtreerpo_segments.meta_info["global_step"] = self.global_steps
                                    # global_token_num needed for MFU calculation in update_actor
                                    # Must be a list of per-sequence token counts
                                    attn_mask = dtreerpo_segments.batch["attention_mask"]
                                    token_counts = attn_mask.sum(dim=-1).long().tolist()
                                    dtreerpo_segments.meta_info["global_token_num"] = token_counts
                                    actor_output = self.actor_rollout_wg.update_actor(dtreerpo_segments)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(actor_output_metrics)

                            # For downstream metrics, create minimal token_level_scores etc.
                            if "token_level_scores" not in batch.batch.keys():
                                n_prompts = batch.batch.batch_size[0]
                                resp_len = self.config.actor_rollout_ref.rollout.get("response_length", 1)
                                dummy_scores = torch.zeros(n_prompts, resp_len)
                                batch.batch["token_level_scores"] = dummy_scores
                                batch.batch["token_level_rewards"] = dummy_scores

                    else:
                        raise NotImplementedError(f"Unsupported algorithm: {self.config.algorithm.name}")

                    if self.config.algorithm.name not in ["mdpo", "dtreerpo"]:
                        # Standard flow for non-MDPO algorithms
                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer("ref", timing_raw):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with _timer("values", timing_raw):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with _timer("adv", timing_raw):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            print(f"{list(reward_extra_infos_dict.keys())=}")
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # compute advantages, executed on the driver process

                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            )

                        # update critic
                        if self.use_critic:
                            with _timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # implement critic warmup
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with _timer("update_actor", timing_raw):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled (not applicable for dtreerpo)
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir and self.config.algorithm.name != "dtreerpo":
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                if self.config.algorithm.name != "dtreerpo":
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    # TODO: implement actual tflpo and theoretical tflpo
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                else:
                    # dtreerpo batch doesn't have standard responses/attention_mask/global_token_num
                    metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
