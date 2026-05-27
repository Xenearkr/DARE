#!/usr/bin/env python3
"""Compare SGLang Dream rollout: train (_per_sample + strip) vs val (batch, no strip).

Uses the same prompt token ids for both paths (mirrors sglang_dream_rollout.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

DARE_ROOT = Path(__file__).resolve().parents[2]
SGLANG_PYTHON = DARE_ROOT / "third_party" / "sglang" / "python"
MASK_TOKEN_ID = 151666
PAD_TOKEN_ID = 151643
DEFAULT_MODEL = DARE_ROOT / "models/finetune_d3LLM"

if str(DARE_ROOT) not in sys.path:
    sys.path.insert(0, str(DARE_ROOT))
if str(SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_PYTHON))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--parquet",
        type=Path,
        default=DARE_ROOT / "data/preprocessed/rl/train/lcbv5-K8_1.parquet",
    )
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--humaneval", action="store_true", help="Use humaneval row instead of lcb train")
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--train-temperature", type=float, default=0.2)
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--mem-fraction-static", type=float, default=0.32)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def load_prompt_ids(tokenizer, parquet: Path, row: int, max_prompt_length: int) -> tuple[list[int], str]:
    import pandas as pd

    df = pd.read_parquet(parquet)
    r = df.iloc[row]
    messages = r.get("prompt")
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if isinstance(messages, str):
        messages = json.loads(messages)
    if hasattr(tokenizer, "apply_chat_template") and messages:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = str(messages)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_prompt_length:
        ids = ids[-max_prompt_length:]
    return ids, text


def build_engine(model_path: Path, max_new_tokens: int, threshold: float, mem_fraction: float):
    from sglang.srt.entrypoints.engine import Engine

    algo_cfg = {
        "threshold": threshold,
        "block_add_threshold": 0.1,
        "decoded_token_threshold": 0.95,
        "block_size": 32,
        "temperature": 0.2,
        "top_p": 0.95,
        "cache_delay_iter": 32,
        "refresh_interval": 10000,
        "early_stop": True,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(algo_cfg, f)
        algo_path = f.name
    return Engine(
        model_path=str(model_path),
        dtype="bfloat16",
        trust_remote_code=True,
        tp_size=1,
        mem_fraction_static=mem_fraction,
        disable_cuda_graph=True,
        attention_backend="torch_native",
        max_running_requests=1,
        dllm_algorithm="FullAttnMultiBlock",
        dllm_algorithm_config=algo_path,
    )


def ids_stats(ids: list[int], label: str) -> dict:
    mask_pos = [i for i, t in enumerate(ids) if t == MASK_TOKEN_ID]
    pad_pos = [i for i, t in enumerate(ids) if t == PAD_TOKEN_ID]
    return {
        "label": label,
        "len": len(ids),
        "n_mask": len(mask_pos),
        "n_pad": len(pad_pos),
        "mask_positions_head": mask_pos[:20],
        "mask_positions_tail": mask_pos[-10:] if len(mask_pos) > 10 else mask_pos,
        "first_mask_at": mask_pos[0] if mask_pos else None,
        "last_mask_at": mask_pos[-1] if mask_pos else None,
    }


def run_train_path(engine, tokenizer, prompt_ids: list[int], args) -> dict:
    from verl.workers.rollout.sglang_rollout.sglang_dream_rollout import _strip_mask_pad_tokens
    from verl.workers.rollout.sglang_rollout.sglang_rollout import _post_process_outputs
    from verl.utils.torch_functional import pad_sequence_to_length

    loop = asyncio.get_event_loop()
    per_call = {
        "n": 1,
        "top_p": 0.95,
        "temperature": args.train_temperature,
        "max_new_tokens": args.max_new_tokens,
        "sampling_seed": args.train_seed,
    }
    out = loop.run_until_complete(
        engine.async_generate(
            prompt=None,
            sampling_params=per_call,
            return_logprob=True,
            input_ids=[prompt_ids],
            image_data=None,
        )
    )
    if isinstance(out, list):
        out = out[0]
    raw_resp, _ = _post_process_outputs(tokenizer, [out])
    raw_ids = raw_resp[0].tolist()
    stripped = _strip_mask_pad_tokens(raw_resp, MASK_TOKEN_ID, PAD_TOKEN_ID)
    if stripped.shape[1] < args.max_new_tokens:
        stripped = pad_sequence_to_length(stripped, args.max_new_tokens, PAD_TOKEN_ID)
    stripped_ids = stripped[0].tolist()
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    return {
        "path": "train_per_sample",
        "sampling": per_call,
        "nfe": meta.get("nfe"),
        "raw": ids_stats(raw_ids, "raw"),
        "after_strip": ids_stats(stripped_ids, "after_strip"),
        "raw_text": tokenizer.decode(raw_ids, skip_special_tokens=True),
        "stripped_text": tokenizer.decode(stripped_ids, skip_special_tokens=True),
    }


def run_val_path(engine, tokenizer, prompt_ids: list[int], args) -> dict:
    from verl.workers.rollout.sglang_rollout.sglang_rollout import _post_process_outputs
    from verl.utils.torch_functional import pad_sequence_to_length

    loop = asyncio.get_event_loop()
    # Mirrors SGLangRollout._batch_level_generate_sequences when is_validate=True
    val_kwargs = {"top_p": 0.95, "temperature": 0.0, "n": 1, "max_new_tokens": args.max_new_tokens}
    out = loop.run_until_complete(
        engine.async_generate(
            prompt=None,
            sampling_params=val_kwargs,
            return_logprob=True,
            input_ids=[prompt_ids],
            image_data=None,
        )
    )
    if isinstance(out, list):
        out_list = out
    else:
        out_list = [out]
    resp, _ = _post_process_outputs(tokenizer, out_list)
    raw_ids = resp[0].tolist()
    if resp.shape[1] < args.max_new_tokens:
        resp = pad_sequence_to_length(resp, args.max_new_tokens, PAD_TOKEN_ID)
    padded_ids = resp[0].tolist()
    meta = out_list[0].get("meta_info", {}) if isinstance(out_list[0], dict) else {}
    return {
        "path": "val_batch",
        "sampling": val_kwargs,
        "nfe": meta.get("nfe"),
        "raw": ids_stats(raw_ids, "raw"),
        "after_pad": ids_stats(padded_ids, "after_pad_no_strip"),
        "raw_text": tokenizer.decode(raw_ids, skip_special_tokens=True),
        "padded_text": tokenizer.decode(padded_ids, skip_special_tokens=True),
    }


def print_section(title: str, body: str, max_chars: int = 2400):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    if len(body) > max_chars:
        half = max_chars // 2
        print(body[:half])
        print("\n... [truncated for display] ...\n")
        print(body[-half:])
    else:
        print(body)


def main():
    args = parse_args()
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    if args.humaneval:
        args.parquet = DARE_ROOT / "data/preprocessed/rl/test/humaneval_1.parquet"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    prompt_ids, prompt_text = load_prompt_ids(tokenizer, args.parquet, args.row, args.max_prompt_length)

    print(f"parquet={args.parquet} row={args.row}")
    print(f"prompt_tokens={len(prompt_ids)} max_new_tokens={args.max_new_tokens}")
    print(f"prompt_tail (last 400 chars):\n{prompt_text[-400:]}")

    engine = build_engine(args.model_path, args.max_new_tokens, args.threshold, args.mem_fraction_static)
    try:
        train_r = run_train_path(engine, tokenizer, prompt_ids, args)
        val_r = run_val_path(engine, tokenizer, prompt_ids, args)
    finally:
        del engine
        torch.cuda.empty_cache()

    result = {
        "prompt_tokens": len(prompt_ids),
        "train": train_r,
        "val": val_r,
        "text_equal_raw": train_r["raw_text"] == val_r["raw_text"],
        "text_equal_decoded": train_r["stripped_text"] == val_r["padded_text"],
    }

    print("\n--- token stats ---")
    print(json.dumps({"train": {k: train_r[k] for k in ("path", "sampling", "nfe", "raw", "after_strip")}}, indent=2))
    print(json.dumps({"val": {k: val_r[k] for k in ("path", "sampling", "nfe", "raw", "after_pad")}}, indent=2))
    print(f"\ntrain_raw == val_raw text: {result['text_equal_raw']}")
    print(f"train_stripped == val_padded text: {result['text_equal_decoded']}")

    print_section("TRAIN raw decode (before strip)", train_r["raw_text"])
    print_section("TRAIN after _strip_mask_pad_tokens", train_r["stripped_text"])
    print_section("VAL raw decode (no strip)", val_r["raw_text"])
    print_section("VAL after pad only", val_r["padded_text"])

    raw = train_r["raw"]
    if raw["n_mask"] > 0 and raw.get("first_mask_at") is not None and raw["first_mask_at"] < raw["len"] - 1:
        print(
            f"\n[note] train raw has {raw['n_mask']} mask tokens; "
            f"first_mask_at={raw['first_mask_at']} last_mask_at={raw['last_mask_at']} "
            "→ strip may splice segments."
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
