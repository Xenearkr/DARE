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

from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.workers.reward_manager.dapo import DAPORewardManager
from verl.utils.reward_score import dllm_rm
from verl.utils.reward_score.code_efficiency import (
    TpfBaselineTracker,
    TpfEfficiencyConfig,
    compute_tpf,
    normalize_rollout_nfe,
)


class DLLMRewardManager(DAPORewardManager):
    """DLLM reward manager with optional TPF-based efficiency shaping for code RL."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        enable_tpf_efficiency: bool = False,
        tpf_efficiency_coef: float = 0.1,
        tpf_baseline_initial: float = 2.0,
        tpf_efficiency_max_bonus: float = 0.25,
        tpf_efficiency_max_penalty: float = 0.25,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score or dllm_rm,
            reward_fn_key=reward_fn_key,
            max_resp_len=max_resp_len,
            overlong_buffer_cfg=overlong_buffer_cfg,
        )
        self._tpf_tracker = TpfBaselineTracker(
            TpfEfficiencyConfig(
                enable=enable_tpf_efficiency,
                coef=tpf_efficiency_coef,
                initial_baseline=tpf_baseline_initial,
                max_bonus=tpf_efficiency_max_bonus,
                max_penalty=tpf_efficiency_max_penalty,
            )
        )

    def _decode_sample(self, data_item):
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        vocab_size = len(self.tokenizer)
        valid_prompt_ids = valid_prompt_ids[valid_prompt_ids < vocab_size]
        valid_response_ids = valid_response_ids[valid_response_ids < vocab_size]
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        eos_token = self.tokenizer.eos_token
        if eos_token and response_str.endswith(eos_token):
            response_str = response_str[: -len(eos_token)]
        return prompt_str, response_str, int(valid_response_length.item())

    def _rollout_stats(self, data_item, valid_response_length: int) -> tuple[int, int, float]:
        ntb = data_item.non_tensor_batch
        nfe = normalize_rollout_nfe(ntb.get("rollout_nfe", 0))
        gen_tokens = ntb.get("rollout_gen_tokens")
        if gen_tokens is None or int(gen_tokens) <= 0:
            gen_tokens = valid_response_length
        else:
            gen_tokens = int(gen_tokens)
        return nfe, gen_tokens, compute_tpf(gen_tokens, nfe)

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        is_validate = bool((data.meta_info or {}).get("validate", False))
        apply_efficiency = self._tpf_tracker.cfg.enable and not is_validate
        baseline = self._tpf_tracker.baseline

        decoded = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_str, response_str, valid_response_length = self._decode_sample(data_item)
            nfe, gen_tokens, tpf = self._rollout_stats(data_item, valid_response_length)
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            if extra_info is None:
                extra_info = {}
            elif hasattr(extra_info, "item"):
                extra_info = extra_info.item()
            extra_info = dict(extra_info)
            extra_info.setdefault("task", "code")
            extra_info["rollout_nfe"] = nfe
            extra_info["rollout_gen_tokens"] = gen_tokens
            extra_info["tpf"] = tpf
            extra_info["tpf_baseline"] = baseline

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            decoded.append(
                {
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "valid_response_length": valid_response_length,
                    "data_source": data_source,
                    "ground_truth": ground_truth,
                    "result": result,
                    "nfe": nfe,
                    "gen_tokens": gen_tokens,
                    "tpf": tpf,
                }
            )

        passed_tpfs = [
            row["tpf"] for row in decoded if row["tpf"] > 0 and self._result_acc(row["result"])
        ]
        baseline = self._tpf_tracker.baseline

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i, row in enumerate(decoded):
            result = row["result"]
            if isinstance(result, dict):
                pass_reward = float(result.get("pass_reward", result.get("score", 0.0)))
                acc = bool(result.get("acc", result.get("is_correct", False)))
            else:
                pass_reward = float(result)
                acc = pass_reward > 0

            efficiency_reward = 0.0
            if apply_efficiency:
                efficiency_reward = self._tpf_tracker.efficiency_reward(row["tpf"], passed=acc)

            score = pass_reward + efficiency_reward

            if isinstance(result, dict):
                _managed_keys = frozenset(
                    {
                        "score",
                        "reward",
                        "pass_reward",
                        "efficiency_reward",
                        "tpf",
                        "rollout_nfe",
                        "rollout_gen_tokens",
                        "tpf_baseline",
                    }
                )
                for key, value in result.items():
                    if key in _managed_keys:
                        continue
                    reward_extra_info[key].append(value)

            reward_extra_info["pass_reward"].append(pass_reward)
            reward_extra_info["efficiency_reward"].append(efficiency_reward)
            reward_extra_info["reward"].append(score)
            reward_extra_info["score"].append(score)
            reward_extra_info["tpf"].append(row["tpf"])
            reward_extra_info["rollout_nfe"].append(row["nfe"])
            reward_extra_info["rollout_gen_tokens"].append(row["gen_tokens"])
            reward_extra_info["tpf_baseline"].append(baseline)

            valid_response_length = row["valid_response_length"]
            reward = score
            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            data_source = row["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", row["prompt_str"])
                print("[response]", row["response_str"])
                print("[ground_truth]", row["ground_truth"])
                print("[pass_reward]", pass_reward)
                print("[efficiency_reward]", efficiency_reward)
                print("[reward]", score)
                print("[tpf]", row["tpf"], "baseline", baseline, "nfe", row["nfe"])

        for tpf in passed_tpfs:
            self._tpf_tracker.observe_passed(tpf)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor

    @staticmethod
    def _result_acc(result) -> bool:
        if isinstance(result, dict):
            if "acc" in result:
                return bool(result["acc"])
            return bool(result.get("is_correct", False))
        return float(result) > 0
