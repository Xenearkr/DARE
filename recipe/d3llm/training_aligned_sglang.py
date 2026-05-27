"""SGLang Engine kwargs aligned with ``SGLangDreamRollout._init_inference_engine``."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DreamSGLangEngineConfig:
    """Defaults mirror ``recipe/dream/run_bgpo_dream_coder_d3llm.sh`` smoke + sglang_dream_rollout."""

    model_path: Path
    rank: int = 0
    mem_fraction_static: float = 0.32
    attention_backend: str = "torch_native"
    disable_cuda_graph: bool = True
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    load_format: str = "auto"
    max_running_requests: int = 1
    tp_size: int = 1
    # Dream / d3LLM algorithm (FullAttnMultiBlock)
    dllm_algorithm: str = "FullAttnMultiBlock"
    threshold: float = 0.5
    block_add_threshold: float = 0.1
    decoded_token_threshold: float = 0.95
    block_size: int = 32
    cache_delay_iter: int = 32
    refresh_interval: int = 10000
    early_stop: bool = True
    train_temperature: float = 0.2
    top_p: float = 0.95
    port_base: int = 30000

    @classmethod
    def smoke_training(cls, model_path: Path, rank: int = 0) -> "DreamSGLangEngineConfig":
        return cls(model_path=model_path, rank=rank)

    @classmethod
    def dedicated_gpu(cls, model_path: Path, rank: int = 0) -> "DreamSGLangEngineConfig":
        """Same algorithm knobs as training; more static mem when GPU is SGLang-only."""
        return cls(model_path=model_path, rank=rank, mem_fraction_static=0.45)


def build_dream_sglang_engine(cfg: DreamSGLangEngineConfig):
    """Create SGLang Engine matching training rollout (tp_size=1 per Ray rank / benchmark worker)."""
    from sglang.srt.entrypoints.engine import Engine

    os.environ.setdefault("SGLANG_BLOCK_NONZERO_RANK_CHILDREN", "0")

    algo_cfg = {
        "threshold": float(cfg.threshold),
        "block_add_threshold": float(cfg.block_add_threshold),
        "decoded_token_threshold": float(cfg.decoded_token_threshold),
        "block_size": int(cfg.block_size),
        "temperature": float(cfg.train_temperature),
        "top_p": float(cfg.top_p),
        "cache_delay_iter": int(cfg.cache_delay_iter),
        "refresh_interval": int(cfg.refresh_interval),
        "early_stop": bool(cfg.early_stop),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(algo_cfg, f)
        algo_path = f.name

    engine_kwargs: dict[str, Any] = dict(
        model_path=str(cfg.model_path),
        dtype=cfg.dtype,
        mem_fraction_static=cfg.mem_fraction_static,
        enable_memory_saver=True,
        base_gpu_id=0,
        gpu_id_step=1,
        tp_size=cfg.tp_size,
        node_rank=0,
        nnodes=1,
        dist_init_addr=None,
        load_format=cfg.load_format,
        trust_remote_code=cfg.trust_remote_code,
        port=cfg.port_base + cfg.rank,
        max_running_requests=cfg.max_running_requests,
        dllm_algorithm=cfg.dllm_algorithm,
        dllm_algorithm_config=algo_path,
        disable_cuda_graph=cfg.disable_cuda_graph,
        attention_backend=cfg.attention_backend,
    )
    return Engine(**engine_kwargs)
