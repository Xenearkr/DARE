#!/usr/bin/env python3
"""Stress multiblock on longest training prompts (512 tokens)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
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


def build_padded_prompt(tok, messages, device):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > MAX_PROMPT:
        ids = ids[-MAX_PROMPT:]
    plen = len(ids)
    padded = torch.full((1, MAX_PROMPT), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros((1, MAX_PROMPT), dtype=torch.long, device=device)
    padded[0, MAX_PROMPT - plen :] = torch.tensor(ids, dtype=torch.long, device=device)
    mask[0, MAX_PROMPT - plen :] = 1
    return padded.to(device), mask.to(device), plen


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    device = "cuda:0"
    df = pd.read_parquet(DARE_ROOT / "data/preprocessed/rl/train/lcbv5-K8_1.parquet")
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        str(MODEL), torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    ).to(device).eval()

    gen_kwargs = {
        "gen_length": RESP_LEN,
        "block_length": 32,
        "temperature": 0.2,
        "do_sample": True,
        "threshold": 0.5,
        "block_add_threshold": 0.1,
        "decoded_token_threshold": 0.95,
        "cache_delay_iter": 32,
        "early_stop": True,
        "per_sample_seed": True,
        "base_seed": 42,
        "pad_token_id": PAD_ID,
    }

    # longest prompts in first 8 rows
    for idx in [0, 4, 2]:
        messages = df.iloc[idx]["prompt"]
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        padded, mask, plen = build_padded_prompt(tok, messages, device)
        batch = padded.repeat(4, 1)
        bmask = mask.repeat(4, 1)
        t0 = time.time()
        with torch.no_grad():
            responses, _, _, _ = execute_dream_multiblock_generation(
                module=model,
                gen_kwargs=gen_kwargs,
                idx_repeat=batch,
                attention_mask_repeat=bmask,
                response_length=RESP_LEN,
                tokenizer=tok,
                process_outputs_fn=process_fastdream_generation_outputs,
            )
        elapsed = time.time() - t0
        lens = [len(tok.decode(r, skip_special_tokens=True)) for r in responses]
        print(f"idx={idx} plen={plen} bs=4 elapsed={elapsed:.1f}s lens={lens}")
        if elapsed > 180:
            print("[WARN] >3min for 4 samples")
            sys.exit(1)
    print("[PASS] long-prompt benchmark")


if __name__ == "__main__":
    main()
