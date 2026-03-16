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
Minimal HF rollout for LLaDA2.0-mini.

Unlike the original LLaDA rollout, this path avoids the packed
`cu_seqlens`-based generation helpers and calls the model's native
block-wise `generate()` method directly.
"""

import re
import time

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from .base import BaseRollout

__all__ = ["DLLMRollout"]


class DLLMRollout(BaseRollout):
    def __init__(self, module: nn.Module, config, tokenizer):
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = tokenizer
        self.MASK_TOKEN_ID = self.module.config.mask_token_id
        self.PAD_TOKEN_ID = self.module.config.pad_token_id
        self.EOS_TOKEN_ID = self.module.config.eos_token_id

        self.response_length = config["response_length"]
        self.num_diffusion_steps = config["num_diffusion_steps"]
        self.block_length = config["block_length"]
        self.n_rollout = config["n"]
        self.temperature = config["temperature"]
        self.val_kwargs = config["val_kwargs"]

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        start_time = time.time()

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        batch_size = idx.size(0)
        is_validate = prompts.meta_info.get("validate", False)
        n_rollout = 1 if is_validate else self.n_rollout

        gen_kwargs = {
            "steps": self.val_kwargs["num_diffusion_steps"] if is_validate else self.num_diffusion_steps,
            "gen_length": self.response_length,
            "block_length": self.block_length,
            "temperature": self.val_kwargs.get("temperature", self.temperature) if is_validate else self.temperature,
            "eos_early_stop": True,
            "eos_id": self.EOS_TOKEN_ID,
            "mask_id": self.MASK_TOKEN_ID,
        }
        print(f"llada2 gen_kwargs: {gen_kwargs}")

        idx_repeat = idx.repeat_interleave(n_rollout, dim=0)
        attention_mask_repeat = attention_mask.repeat_interleave(n_rollout, dim=0)
        total_batch_size = batch_size * n_rollout

        responses = []
        full_attention_masks = []
        answers = []

        self.module.eval()
        for i in range(total_batch_size):
            valid_prompt = idx_repeat[i][attention_mask_repeat[i].bool()].unsqueeze(0)
            output = self.module.generate(inputs=valid_prompt, **gen_kwargs)
            output = output[:, : self.response_length]

            padded_response = torch.full((1, self.response_length), self.PAD_TOKEN_ID, device=idx.device, dtype=idx.dtype)
            padded_response[:, : output.size(1)] = output
            responses.append(padded_response)

            response_mask = torch.zeros((1, self.response_length), device=attention_mask.device, dtype=attention_mask.dtype)
            response_mask[:, : output.size(1)] = 1
            full_attention_masks.append(torch.cat([attention_mask_repeat[i : i + 1], response_mask], dim=1))

            response_str = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            boxed_matches = re.findall(r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}", response_str, re.DOTALL)
            answer_matches = re.findall(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
            answers.append(list(set(boxed_matches + answer_matches)))

        responses_cat = torch.cat(responses, dim=0)
        attention_masks_cat = torch.cat(full_attention_masks, dim=0)
        input_ids_cat = torch.cat([idx_repeat, responses_cat], dim=1)

        try:
            rank = dist.get_rank()
        except Exception:
            rank = 0

        for i, answer in enumerate(answers):
            split = "validation" if is_validate else "rollout"
            print(f"==================[RANK{rank}] {split} question ID: {i}=================\nGenerated answer: {answer}\n==========================================")

        batch = TensorDict(
            {
                "prompts": idx_repeat,
                "responses": responses_cat,
                "input_ids": input_ids_cat,
                "attention_mask": attention_masks_cat,
                "position_ids": torch.cat(
                    [
                        position_ids.repeat_interleave(n_rollout, dim=0),
                        position_ids[:, -1:].repeat_interleave(n_rollout, dim=0)
                        + torch.arange(1, self.response_length + 1, device=position_ids.device),
                    ],
                    dim=1,
                ),
            },
            batch_size=total_batch_size,
        )

        self.module.train()
        print(f"[RANK{rank}] llada2 generate_sequences total time: {(time.time() - start_time):.2f}s")
        return DataProto(batch=batch)
