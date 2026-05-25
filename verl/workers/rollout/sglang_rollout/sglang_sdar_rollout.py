# Copyright 2025 Shanghai AI Lab Ltd. and/or its affiliates
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
SGLang Rollout Worker for SDAR (Block Diffusion) models.

This module provides SGLang-based rollout functionality for SDAR models,
which use block-level diffusion rather than token-level autoregressive generation.
It integrates with SGLang's built-in dLLM (diffusion LLM) support.
"""

import asyncio
import logging
import os
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.sglang_rollout.sglang_rollout import (
    SGLangRollout,
    _post_process_outputs,
    _pre_process_inputs,
)
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SGLangSDARRollout(SGLangRollout):
    """SGLang rollout worker for SDAR (block diffusion) models.

    This class extends SGLangRollout to support SDAR's block diffusion forward pass.
    It leverages SGLang's built-in dLLM framework for efficient inference.

    SDAR-specific parameters:
        - mask_token_id: Token ID used for masking (default: 151669 for 8B model)
        - block_length: Block size for diffusion (default: 4)
        - dllm_algorithm: dLLM algorithm to use (e.g., "LowConfidence", "JointThreshold")

    Example config:
        ```yaml
        actor_rollout_ref:
          rollout:
            name: "sglang"
            model_name: "sdar"
            mask_token_id: 151669
            block_length: 4
            dllm_algorithm: "LowConfidence"
            mem_fraction_static: 0.6
            max_running_requests: 1
            attention_backend: "flashinfer"
        ```
    """

    def __init__(
        self,
        actor_module: str,
        config: DictConfig,
        tokenizer: "PreTrainedTokenizer",
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Initialize SGLangSDARRollout.

        Args:
            actor_module: Path to the SDAR model (HuggingFace format)
            config: Configuration DictConfig containing:
                - mask_token_id: SDAR mask token ID (default: 151669)
                - block_length: Block size for diffusion (default: 4)
                - dllm_algorithm: dLLM algorithm (default: "LowConfidence")
                - mem_fraction_static: GPU memory fraction (default: 0.6)
                - max_running_requests: Max concurrent requests (default: 1)
                - attention_backend: Attention backend (default: "flashinfer")
            tokenizer: Tokenizer compatible with the model
            model_hf_config: HuggingFace model configuration
            port: Port for multi-node setup
            trust_remote_code: Whether to trust remote code
            device_mesh: Device mesh for distributed setup
            **kwargs: Additional arguments (e.g., train_tp for Megatron)
        """
        # Store SDAR-specific parameters before calling parent init
        self._mask_token_id = config.get("mask_token_id", 151669)
        self._block_length = config.get("block_length", 4)
        self._dllm_algorithm = config.get("dllm_algorithm", "LowConfidence")
        self._dllm_confidence_threshold = config.get("dllm_confidence_threshold", 0.9)
        self._num_diffusion_steps = config.get("num_diffusion_steps", 4)

        logger.info(
            f"Initializing SGLangSDARRollout with: "
            f"mask_token_id={self._mask_token_id}, "
            f"block_length={self._block_length}, "
            f"dllm_algorithm={self._dllm_algorithm}, "
            f"dllm_confidence_threshold={self._dllm_confidence_threshold}, "
            f"num_diffusion_steps={self._num_diffusion_steps}"
        )

        # Call parent init
        super().__init__(
            actor_module=actor_module,
            config=config,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            port=port,
            trust_remote_code=trust_remote_code,
            device_mesh=device_mesh,
            **kwargs,
        )

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        """Initialize SGLang engine with SDAR/dLLM support.

        This method overrides the parent to add SDAR-specific parameters:
        - dllm_algorithm: The dLLM algorithm to use
        - These parameters are passed to SGLang's Engine initialization

        SGLang will automatically detect the SDAR model architecture and apply
        the appropriate block diffusion forward pass.
        """
        import tempfile

        import yaml
        from sglang.srt.entrypoints.engine import Engine
        from .utils import broadcast_pyobj, get_ip, get_open_port

        # Initialize distributed environment
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group(self._tp_mesh_name),
                src=self._device_mesh_cpu[self._tp_mesh_name].mesh[0].item(),
                force_cpu_device=False,
            )
            from verl.workers.rollout.sglang_rollout.utils import is_ipv6
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            rank = self._rank
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

            # Build engine arguments with SDAR/dLLM specific parameters
            engine_kwargs = dict(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.get("mem_fraction_static", 0.6),
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                port=30000 + rank,
                max_running_requests=self.config.get("max_running_requests", 1),
            )

            # Add SDAR/dLLM specific parameters (read by scheduler subprocess)
            engine_kwargs["dllm_algorithm"] = self._dllm_algorithm
            algo_cfg = {
                "threshold": self._dllm_confidence_threshold,
                "denoising_steps": self._num_diffusion_steps,
                "temperature": float(self.config.get("temperature", 1.0)),
                "top_k": 0,
                "top_p": float(self.config.get("top_p", 1.0)),
            }
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(algo_cfg, f)
                engine_kwargs["dllm_algorithm_config"] = f.name

            if self.config.get("disable_cuda_graph") is not None:
                engine_kwargs["disable_cuda_graph"] = self.config.get("disable_cuda_graph")

            # Optionally add attention backend if specified
            if self.config.get("attention_backend"):
                engine_kwargs["attention_backend"] = self.config.get("attention_backend")

            logger.info(f"Initializing SGLang Engine for SDAR with dllm_algorithm={self._dllm_algorithm}")

            self._engine = Engine(**engine_kwargs)
        else:
            self._engine = None

        self.sharding_manager = None
        if self._tp_rank == 0:
            self._engine.release_memory_occupation()
        self.is_sleep = True

    @property
    def mask_token_id(self) -> int:
        """Return the SDAR mask token ID."""
        return self._mask_token_id

    @property
    def block_length(self) -> int:
        """Return the SDAR block length."""
        return self._block_length

    @property
    def dllm_algorithm(self) -> str:
        """Return the dLLM algorithm being used."""
        return self._dllm_algorithm

    @torch.no_grad()
    def _batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """LMDeploy-style rollout: one independent ``n=1`` generate per sample."""
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample or is_validate:
            return super()._batch_level_generate_sequences(prompts, **kwargs)
        return self._lmdeploy_style_batch_level_generate_sequences(prompts, **kwargs)

    @torch.no_grad()
    def _lmdeploy_style_batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        eos_token_id = prompts.meta_info.get("eos_token_id", self.eos_token_id)
        if eos_token_id is None:
            eos_token_id = self.pad_token_id

        batch_size = idx.size(0)
        n_rollout = self.config.n

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )

        raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        image_data = non_tensor_batch.pop("multi_modal_data", None) if "multi_modal_data" in non_tensor_batch else None

        idx_list = []
        image_list = []
        for i in range(batch_size):
            prompt_ids = raw_prompt_ids[i]
            if isinstance(prompt_ids, np.ndarray):
                prompt_ids = prompt_ids.tolist()
            for _ in range(n_rollout):
                idx_list.append(prompt_ids)
                if image_data is not None:
                    item = image_data[i]
                    image_list.append(item.get("image", None) if isinstance(item, dict) else None)
                else:
                    image_list.append(None)

        idx_repeat = idx.repeat_interleave(n_rollout, dim=0)
        attention_mask_repeat = attention_mask.repeat_interleave(n_rollout, dim=0)
        position_ids_repeat = position_ids.repeat_interleave(n_rollout, dim=0)
        total_batch_size = batch_size * n_rollout

        _non_tensor_batch = {}
        for key, val in non_tensor_batch.items():
            _non_tensor_batch[key] = np.repeat(val, n_rollout, axis=0)

        gen_kwargs = dict(
            n=1,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            max_new_tokens=self.config.response_length,
        )

        if self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            merged_output = []
            for input_ids, image in zip(idx_list, image_list):
                per_call_params = {
                    **self.sampling_params,
                    **gen_kwargs,
                    "sampling_seed": random.randint(0, 2**31 - 1),
                }
                output = loop.run_until_complete(
                    self._engine.async_generate(
                        prompt=None,
                        sampling_params=per_call_params,
                        return_logprob=True,
                        input_ids=[input_ids],
                        image_data=[image] if image is not None else None,
                    )
                )
                if isinstance(output, list):
                    merged_output.extend(output)
                else:
                    merged_output.append(output)
        else:
            merged_output = None

        [merged_output] = broadcast_pyobj(
            data=[merged_output],
            rank=self._rank,
            dist_group=self._device_mesh_cpu[self._tp_mesh_name].get_group(),
            src=self._device_mesh_cpu[self._tp_mesh_name].mesh[0].item(),
            force_cpu_device=False,
        )

        response, rollout_log_probs = _post_process_outputs(self.tokenizer, merged_output)
        response = response.to(idx.device)
        if rollout_log_probs is not None:
            rollout_log_probs = rollout_log_probs.to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            if rollout_log_probs is not None:
                rollout_log_probs = pad_sequence_to_length(
                    rollout_log_probs, self.config.response_length, 0.0
                )

        seq = torch.cat([idx_repeat, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids_repeat.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(total_batch_size, 1)
        response_position_ids = position_ids_repeat[:, -1:] + delta_position_id
        position_ids_out = torch.cat([position_ids_repeat, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask_repeat.dtype
        )
        attention_mask_out = torch.cat((attention_mask_repeat, response_attention_mask), dim=-1)

        batch_tensors = {
            "prompts": idx_repeat,
            "responses": response,
            "input_ids": seq,
            "attention_mask": attention_mask_out,
            "position_ids": position_ids_out,
        }
        if rollout_log_probs is not None:
            batch_tensors["rollout_log_probs"] = rollout_log_probs

        batch = TensorDict(batch_tensors, batch_size=total_batch_size)

        if self.config.free_cache_engine and self._engine is not None:
            self._engine.flush_cache()

        return DataProto(batch=batch, non_tensor_batch=_non_tensor_batch)
