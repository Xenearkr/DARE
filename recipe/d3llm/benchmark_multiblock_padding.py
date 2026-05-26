#!/usr/bin/env python3
"""Benchmark multiblock: left-padded prompts, batch=4 must match batch=1 quality."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

DARE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DARE_ROOT))

from verl.workers.rollout.dream_multiblock import execute_dream_multiblock_generation
from verl.workers.rollout.rollout_utils import process_fastdream_generation_outputs

MODEL = DARE_ROOT / "models" / "finetune_d3LLM"
MAX_PROMPT = 512
RESP_LEN = 256
PAD_ID = 151643

CODE_PROMPT = """<|im_start|>user
Write a Python function `add(a, b)` that returns the sum of two integers.
<|im_start|>assistant
"""


def left_pad(input_ids: torch.Tensor, attention_mask: torch.Tensor, width: int):
    b, l = input_ids.shape
    out = torch.full((b, width), PAD_ID, dtype=input_ids.dtype, device=input_ids.device)
    mask = torch.zeros((b, width), dtype=attention_mask.dtype, device=attention_mask.device)
    out[:, width - l :] = input_ids
    mask[:, width - l :] = attention_mask
    return out, mask


def run_batch(model, tok, device, batch_ids, batch_mask, gen_kwargs):
    t0 = time.time()
    with torch.no_grad():
        responses, _, _, _ = execute_dream_multiblock_generation(
            module=model,
            gen_kwargs=gen_kwargs,
            idx_repeat=batch_ids,
            attention_mask_repeat=batch_mask,
            response_length=RESP_LEN,
            tokenizer=tok,
            process_outputs_fn=process_fastdream_generation_outputs,
        )
    elapsed = time.time() - t0
    texts = [tok.decode(r, skip_special_tokens=True) for r in responses]
    return elapsed, texts


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(device).eval()

    natural = tok(CODE_PROMPT, return_tensors="pt").to(device)
    padded_ids, padded_mask = left_pad(natural["input_ids"], natural["attention_mask"], MAX_PROMPT)

    gen_kwargs = {
        "gen_length": RESP_LEN,
        "block_length": 32,
        "temperature": 0.0,
        "do_sample": False,
        "threshold": 0.5,
        "block_add_threshold": 0.1,
        "decoded_token_threshold": 0.95,
        "cache_delay_iter": 32,
        "early_stop": True,
        "per_sample_seed": False,
        "pad_token_id": PAD_ID,
    }

    t1, texts1 = run_batch(model, tok, device, padded_ids, padded_mask, gen_kwargs)
    batch_ids = padded_ids.repeat(4, 1)
    batch_mask = padded_mask.repeat(4, 1)
    t4, texts4 = run_batch(model, tok, device, batch_ids, batch_mask, gen_kwargs)

    print(f"bs=1: {t1:.2f}s len0={len(texts1[0])}")
    print(f"bs=4: {t4:.2f}s")
    for i, t in enumerate(texts4):
        print(f"  sample[{i}] len={len(t)} tail={t[-120:]!r}")

    failed = []
    for i, t in enumerate(texts4):
        if len(t) < 50 or "def add" not in t:
            failed.append(i)
    if failed:
        print(f"[FAIL] samples {failed} look truncated or wrong")
        sys.exit(1)
    if t4 > 60:
        print(f"[WARN] bs=4 took {t4:.1f}s (>60s) on single GPU")
    print("[PASS] batch-4 quality ok")


if __name__ == "__main__":
    main()
