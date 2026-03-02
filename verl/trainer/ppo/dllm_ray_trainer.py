from verl.trainer.ppo.ray_trainer import *
from verl.trainer.ppo.ray_trainer import _timer
from verl.trainer.ppo.dllm_metric_utils import (
    process_validation_metrics,
)


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
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

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

                    if self.config.algorithm.name in ["d1", "coupled-grpo", "bgpo", "spg"]:
                        if self.config.actor_rollout_ref.model.name != "sdar":
                            with _timer("forward_process", timing_raw):
                                forward_batch_output = self.actor_rollout_wg.forward_process(batch)
                            batch = batch.union(forward_batch_output)
                        if self.config.algorithm.name in ["bgpo"]:
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
                            response_masks = batch.batch["reversed_traj_unmask_positions"]
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
                                    non_tensor_batch=batch.non_tensor_batch,
                                )
                                reward_t, _ = compute_reward(t_batch, self.reward_fn)
                                step_reward = reward_t.sum(dim=-1)  # (batch_size,)
                                all_step_rewards.append(step_reward)

                            all_step_rewards_tensor = torch.stack(all_step_rewards, dim=-1)  # (batch_size, steps)

                        with _timer("mdpo_advantages", timing_raw):
                            # Compute step-wise advantages
                            num_generations = self.config.actor_rollout_ref.rollout.n
                            step_advantages = compute_step_wise_advantage(
                                all_step_rewards_tensor, num_generations
                            )  # (batch_size, steps)

                            # Select top-K steps for training
                            sample_train_steps = self.config.actor_rollout_ref.actor.get("sample_train_steps", 16)
                            top_k_indices = select_top_k_steps(step_advantages, k=sample_train_steps)
                            print(f"MDPO: selected {len(top_k_indices)} steps for training: {top_k_indices.tolist()}")

                            # Compute final-step reward for standard metrics
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor
                            batch.batch["token_level_rewards"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # Log MDPO-specific metrics
                            metrics["mdpo/mean_step_reward"] = all_step_rewards_tensor.mean().item()
                            metrics["mdpo/final_step_reward"] = all_step_rewards_tensor[:, -1].mean().item()
                            metrics["mdpo/mean_advantage"] = step_advantages.mean().item()
                            metrics["mdpo/num_train_steps"] = len(top_k_indices)

                        # For each selected step, pack data for actor update
                        with _timer("mdpo_pack_steps", timing_raw):
                            all_steps_input_ids = batch.batch["all_steps_input_ids"]  # (batch, steps, response_len)
                            all_confidence = batch.batch["all_confidence"]  # (batch, steps, response_len)
                            attention_mask = batch.batch["attention_mask"]  # (batch, seq_len)

                            # Get mask_token_id from rollout meta_info (set by MDPO rollout)
                            mask_token_id = batch.meta_info.get("mask_token_id", None)

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
                                if mask_token_id is not None:
                                    completion_mask = (all_steps_input_ids[:, step_idx, :] == mask_token_id).float()
                                else:
                                    # Fallback: use response mask from attention mask
                                    resp_attn = attention_mask[:, -response_len:]
                                    completion_mask = resp_attn.float()
                                # Ensure at least some tokens are masked for meaningful gradient
                                if completion_mask.sum() == 0:
                                    completion_mask = attention_mask[:, -response_len:].float()

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

                        # Build the final batch for actor update
                        # Repeat non_tensor_batch to match the concatenated step dimension
                        num_repeats = len(top_k_indices)
                        mdpo_non_tensor_batch = {}
                        for k, v in batch.non_tensor_batch.items():
                            if isinstance(v, np.ndarray):
                                mdpo_non_tensor_batch[k] = np.repeat(v, num_repeats, axis=0)
                            elif isinstance(v, list):
                                mdpo_non_tensor_batch[k] = v * num_repeats
                            else:
                                mdpo_non_tensor_batch[k] = v

                        batch = DataProto.from_dict(
                            tensors={
                                "mdpo_step_input_ids": mdpo_step_input_ids,
                                "mdpo_step_target_ids": mdpo_step_target_ids,
                                "attention_mask": mdpo_attention_mask,
                                "completion_mask": mdpo_completion_mask,
                                "advantages": mdpo_advantages,
                                "confidence": mdpo_confidence,
                                "old_per_token_logps": old_per_token_logps,
                                # Keep these for metrics/logging
                                "prompts": prompts_ids.repeat(num_repeats, 1),
                                "responses": all_steps_completion_ids[:, -1, :].repeat(num_repeats, 1),
                                "token_level_scores": reward_tensor.repeat(num_repeats, 1),
                                "token_level_rewards": reward_tensor.repeat(num_repeats, 1),
                            },
                            non_tensor_batch=mdpo_non_tensor_batch,
                        )
                        batch.batch["response_mask"] = mdpo_completion_mask
                        # Set global_token_num for throughput metrics
                        batch.meta_info["global_token_num"] = torch.sum(mdpo_attention_mask, dim=-1).tolist()

                        # Skip standard forward_process, compute_log_prob, compute_advantage
                        # Go directly to actor update
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with _timer("update_actor", timing_raw):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                    else:
                        raise NotImplementedError(f"Unsupported algorithm: {self.config.algorithm.name}")

                    if self.config.algorithm.name != "mdpo":
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

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
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
                if self.config.algorithm.name != "mdpo":
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
