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
FSDP Sharding Manager for SDAR models with SGLang inference engine.

This module provides weight synchronization between FSDP training process
and SGLang inference engine for SDAR (block diffusion) models.
"""

import logging
import os

import torch
import torch.distributed as dist
from sglang.srt.entrypoints.engine import Engine
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available

from .fsdp_sglang import FSDPSGLangShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FSDPSGLangSDARShardingManager(FSDPSGLangShardingManager):
    """FSDP sharding manager for SDAR models with SGLang inference engine.

    This class extends FSDPSGLangShardingManager to handle SDAR-specific
    weight synchronization requirements. SDAR models have additional
    parameters related to block diffusion that need proper handling.

    Key differences from base FSDPSGLangShardingManager:
        - Handles SDAR's mask_token_id and block_length parameters
        - Ensures block diffusion state is properly synchronized
    """

    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
    ):
        """Initialize FSDPSGLangSDARShardingManager.

        Args:
            module: FSDP-wrapped SDAR model from training process
            inference_engine: SGLang Engine for SDAR inference
            model_config: Model configuration containing SDAR-specific parameters
            full_params: Whether to use full state dict (for single GPU)
            device_mesh: Device mesh for distributed setup
            offload_param: Whether to offload parameters to CPU
        """
        super().__init__(
            module=module,
            inference_engine=inference_engine,
            model_config=model_config,
            full_params=full_params,
            device_mesh=device_mesh,
            offload_param=offload_param,
        )

        # SDAR-specific parameters
        self.mask_token_id = getattr(model_config, 'mask_token_id', 151669)
        self.block_length = getattr(model_config, 'block_length', 4)

        logger.info(
            f"Initialized FSDPSGLangSDARShardingManager with "
            f"mask_token_id={self.mask_token_id}, block_length={self.block_length}"
        )

    @GPUMemoryLogger(role="FSDPSGLangSDARShardingManager enter", logger=logger)
    def __enter__(self):
        """Enter context for weight synchronization.

        This method is called before rollout generation. It:
        1. Loads FSDP model to GPU if offloaded
        2. Gets state dict from FSDP
        3. Updates SGLang engine weights
        4. Optionally offloads FSDP back to CPU

        For SDAR models, this ensures block diffusion parameters are properly synced.
        """
        torch.cuda.empty_cache()
        log_gpu_memory_usage("Before state_dict() in SDAR sharding manager", logger=logger)

        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)

        # Get state dict from FSDP module
        params = self.module.state_dict()

        log_gpu_memory_usage("After state_dict() in SDAR sharding manager", logger=logger)

        device = torch.cuda.current_device()
        params = {
            k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v
            for k, v in params.items()
        }

        # Update SGLang engine weights
        self.update_weights(params)

        log_gpu_memory_usage("After sync SDAR model weights in sharding manager", logger=logger)

        del params
        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)

        torch.cuda.empty_cache()
        log_gpu_memory_usage("After del state_dict and empty_cache in SDAR sharding manager", logger=logger)

        # Set random states for consistency
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangSDARShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context after rollout generation.

        This method:
        1. Releases SGLang engine memory
        2. Sets module back to training mode
        3. Restores random states
        """
        log_gpu_memory_usage("Before SGLang SDAR offload in sharding manager", logger=logger)
        self.release_memory()
        log_gpu_memory_usage("After SGLang SDAR offload in sharding manager", logger=logger)

        self.module.train()
        torch.cuda.empty_cache()

        # Restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
