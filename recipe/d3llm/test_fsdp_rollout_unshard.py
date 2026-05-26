#!/usr/bin/env python3
"""Verify FSDP2 unshard allows desync inference without deadlock.

Usage (4 GPUs):
  torchrun --nproc_per_node=4 recipe/d3llm/test_fsdp_rollout_unshard.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

DARE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DARE_ROOT))

from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict, fsdp_rollout_inference_context
from verl.workers.rollout.dream_multiblock import execute_dream_multiblock_generation
from verl.workers.rollout.rollout_utils import process_fastdream_generation_outputs
from transformers import AutoModel, AutoTokenizer


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model_path = DARE_ROOT / "models" / "finetune_d3LLM"
    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(device)

    from torch.distributed.device_mesh import init_device_mesh
    from verl.utils.fsdp_utils import MixedPrecisionPolicy

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
    fsdp_kwargs = {"mesh": mesh, "mp_policy": mp, "offload_policy": None, "reshard_after_forward": True}
    full_state = model.state_dict()
    fsdp_config = {"wrap_policy": {"transformer_layer_cls_to_wrap": ["DreamDecoderLayer"]}}
    apply_fsdp2(model, fsdp_kwargs, fsdp_config)
    fsdp2_load_full_state_dict(model, full_state, mesh, None)
    model.eval()

    # Different prompt lengths per rank to force different NFE.
    prompts = [
        "Write `add(a,b)` in Python.",
        "Write `mul(a,b)` in Python with detailed explanation and edge cases.",
        "Write `sub(a,b)` in Python.",
        "Write `div(a,b)` in Python with docstring and type hints and examples.",
    ]
    text = f"<|im_start|>user\n{prompts[rank % 4]}\n<|im_start|>assistant\n"
    enc = tok(text, return_tensors="pt")
    pad_id = 151643
    width = 512
    plen = enc["input_ids"].size(1)
    ids = torch.full((1, width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((1, width), dtype=torch.long, device=device)
    ids[0, width - plen :] = enc["input_ids"][0].to(device)
    mask[0, width - plen :] = 1

    gen_kwargs = {
        "gen_length": 128,
        "block_length": 32,
        "temperature": 0.0,
        "do_sample": False,
        "threshold": 0.5,
        "block_add_threshold": 0.1,
        "decoded_token_threshold": 0.95,
        "cache_delay_iter": 32,
        "early_stop": True,
        "per_sample_seed": False,
        "pad_token_id": pad_id,
    }

    t0 = time.time()
    with fsdp_rollout_inference_context(model):
        with torch.no_grad():
            responses, _, _, _ = execute_dream_multiblock_generation(
                module=model,
                gen_kwargs=gen_kwargs,
                idx_repeat=ids,
                attention_mask_repeat=mask,
                response_length=128,
                tokenizer=tok,
                process_outputs_fn=process_fastdream_generation_outputs,
            )
    elapsed = time.time() - t0
    text_out = tok.decode(responses[0], skip_special_tokens=True)
    print(
        f"[RANK{rank}] PASS elapsed={elapsed:.1f}s plen={plen} out_len={len(text_out)}",
        flush=True,
    )
    dist.barrier()
    if rank == 0:
        print("[PASS] all ranks finished fsdp2 desync rollout test", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
