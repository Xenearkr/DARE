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
LLaDA2 block-diffusion SFT dataset.

This keeps the original clean sequence and the noised sequence at the same time
so the trainer can build the official `[noisy_x, clean_x]` block-diffusion input.
"""

import numpy as np
import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class dLLMSFTDataset(Dataset):
    def __init__(self, parquet_files: str | ListConfig, tokenizer, config, eval=False, max_samples: int = -1):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.mask_token_id = config.get("mask_token_id", 156895)
        self.noise_range_low = config.get("noise_range_low", 0.3)
        self.noise_range_high = config.get("noise_range_high", 0.8)
        self.eval = eval

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.max_samples = max_samples
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []
        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

        if self.eval:
            steps = len(self.prompts)
            self.t = torch.linspace(
                self.noise_range_low,
                self.noise_range_high,
                steps=steps,
                dtype=torch.float32,
            )

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframes.append(pd.read_parquet(parquet_file))
        self.dataframe = pd.concat(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()

        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

    def _forward_process(self, input_ids, attention_mask, prompt_length, item):
        device = input_ids.device
        sequence_length = input_ids.shape[0]

        if self.eval:
            t = self.t[item].to(device)
        else:
            t = torch.rand((), device=device)
            t = self.noise_range_low + (self.noise_range_high - self.noise_range_low) * t

        valid_mask = attention_mask.bool()
        mask_indices = torch.rand((sequence_length,), device=device) < t
        mask_indices[:prompt_length] = False
        mask_indices &= valid_mask

        noisy_ids = input_ids.clone()
        noisy_ids[mask_indices] = self.mask_token_id

        labels = input_ids.clone()
        labels[~mask_indices] = -100
        labels[:prompt_length] = -100
        return noisy_ids, t, mask_indices, labels

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        prompt = self.prompts[item]
        response = self.responses[item]

        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_length = self.max_length - sequence_length
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            padded_input_ids = torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros((pad_length,), dtype=attention_mask.dtype)
            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)
        clean_input_ids = input_ids.clone()
        noisy_input_ids, t, mask_indices, labels = self._forward_process(
            clean_input_ids, attention_mask, prompt_length, item
        )

        return {
            "input_ids": noisy_input_ids,
            "noisy_input_ids": noisy_input_ids,
            "clean_input_ids": clean_input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": mask_indices,
            "t": t,
            "labels": labels,
        }
