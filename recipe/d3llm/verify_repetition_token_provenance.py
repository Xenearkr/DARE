#!/usr/bin/env python3
"""Prove repetition is in SGLang output_ids, not introduced by VERL postprocess."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

DARE_ROOT = Path(__file__).resolve().parents[2]
SGLANG_PYTHON = DARE_ROOT / "third_party" / "sglang" / "python"
MASK_TOKEN_ID = 151666
PAD_TOKEN_ID = 151643

if str(DARE_ROOT) not in sys.path:
    sys.path.insert(0, str(DARE_ROOT))
if str(SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_PYTHON))


def load_prompt_ids(tokenizer, row: int = 5) -> list[int]:
    import pandas as pd

    df = pd.read_parquet(DARE_ROOT / "data/preprocessed/rl/train/lcbv5-K8_1.parquet")
    m = df.iloc[row]["prompt"]
    if hasattr(m, "tolist"):
        m = m.tolist()
    text = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[-1024:] if len(ids) > 1024 else ids


def main():
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    from sglang.srt.entrypoints.engine import Engine
    from transformers import AutoTokenizer

    from verl.workers.rollout.sglang_rollout.sglang_rollout import _post_process_outputs

    model_path = DARE_ROOT / "models/finetune_d3LLM"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    prompt_ids = load_prompt_ids(tokenizer, row=5)

    algo_cfg = {
        "threshold": 0.5,
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

    engine = Engine(
        model_path=str(model_path),
        dtype="bfloat16",
        trust_remote_code=True,
        tp_size=1,
        mem_fraction_static=0.32,
        disable_cuda_graph=True,
        attention_backend="torch_native",
        max_running_requests=1,
        dllm_algorithm="FullAttnMultiBlock",
        dllm_algorithm_config=algo_path,
    )

    sampling = {
        "n": 1,
        "top_p": 0.95,
        "temperature": 0.2,
        "max_new_tokens": 512,
        "sampling_seed": 42,
    }
    loop = asyncio.get_event_loop()
    resp = loop.run_until_complete(
        engine.async_generate(
            prompt=None,
            sampling_params=sampling,
            return_logprob=True,
            input_ids=[prompt_ids],
        )
    )
    if isinstance(resp, list):
        resp = resp[0]

    # --- every id-bearing field in the raw SGLang response ---
    meta = resp.get("meta_info") or {}
    engine_output_ids = list(resp.get("output_ids") or [])
    engine_token_ids = list(resp.get("token_ids") or [])
    engine_text = resp.get("text") or ""

    logprob_ids = []
    otlp = meta.get("output_token_logprobs") or []
    for item in otlp:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            logprob_ids.append(int(item[1]))

    verl_tensor, _ = _post_process_outputs(tokenizer, [resp])
    verl_ids = verl_tensor[0].tolist()

    # decode each source independently (no strip, no pad)
    def dec(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    sources = {
        "resp.output_ids": engine_output_ids,
        "resp.token_ids": engine_token_ids,
        "meta.output_token_logprobs[:,1]": logprob_ids,
        "verl._post_process_outputs": verl_ids,
    }

    print("=== response keys ===")
    print(sorted(resp.keys()))
    print("meta keys:", sorted(meta.keys()))

    print("\n=== length / equality ===")
    for name, ids in sources.items():
        print(f"{name}: len={len(ids)}")
    if engine_output_ids:
        print("output_ids == verl_ids:", engine_output_ids == verl_ids)
    if logprob_ids:
        print("logprob_ids == verl_ids:", logprob_ids == verl_ids)
        print("logprob_ids == output_ids:", logprob_ids == engine_output_ids)
    if engine_token_ids:
        print("token_ids == output_ids:", engine_token_ids == engine_output_ids)

    print("\n=== engine resp.text vs decode(output_ids) ===")
    text_from_output_ids = dec(engine_output_ids) if engine_output_ids else ""
    print("resp.text == decode(output_ids):", engine_text == text_from_output_ids)
    print("verl decode == decode(output_ids):", dec(verl_ids) == text_from_output_ids)

    # locate 2nd occurrence of opening phrase in token stream
    phrase = "To solve this problem"
    phrase_ids = tokenizer.encode(phrase, add_special_tokens=False)

    def find_all_subseq(haystack: list[int], needle: list[int]) -> list[int]:
        hits = []
        n = len(needle)
        for i in range(len(haystack) - n + 1):
            if haystack[i : i + n] == needle:
                hits.append(i)
        return hits

    ids_for_search = engine_output_ids or verl_ids
    hits = find_all_subseq(ids_for_search, phrase_ids)
    print(f"\n=== phrase {phrase!r} token ids {phrase_ids} ===")
    print(f"occurrences at token indices: {hits}")

    if len(hits) >= 2:
        i0, i1 = hits[0], hits[1]
        # show boundary before 2nd occurrence: last 25 tokens before i1
        boundary_start = max(0, i1 - 25)
        seg = ids_for_search[boundary_start : i1 + len(phrase_ids) + 5]
        print(f"\n=== tokens [{boundary_start}:{i1 + len(phrase_ids) + 5}] (around 2nd phrase) ===")
        print("ids:", seg)
        print("decode:", repr(dec(seg)))
        print("\n=== prove: decode(ids[:i1]) ends with, decode(ids[i1:]) starts with phrase ===")
        prefix = dec(ids_for_search[:i1])
        suffix = dec(ids_for_search[i1:])
        print("decode(ids[:i1]) tail (80 chars):", repr(prefix[-80:]))
        print("decode(ids[i1:]) head (80 chars):", repr(suffix[:80]))

    # splice character offset in full decode
    full = dec(ids_for_search)
    char_hits = []
    start = 0
    while True:
        j = full.find(phrase, start)
        if j < 0:
            break
        char_hits.append(j)
        start = j + 1
    print(f"\n=== character positions of phrase in decode(output_ids): {char_hits} ===")
    if len(char_hits) >= 2:
        c = char_hits[1]
        print("context at 2nd char hit:", repr(full[c - 40 : c + len(phrase) + 40]))

    out_path = Path("/tmp/repetition_token_provenance.json")
    out_path.write_text(
        json.dumps(
            {
                "prompt_tokens": len(prompt_ids),
                "output_ids_len": len(engine_output_ids),
                "phrase_token_positions": hits,
                "phrase_char_positions": char_hits,
                "sources_equal": {
                    "output_ids_eq_verl": engine_output_ids == verl_ids if engine_output_ids else None,
                    "logprob_ids_eq_verl": logprob_ids == verl_ids if logprob_ids else None,
                },
                "output_ids_tail_30": engine_output_ids[-30:] if engine_output_ids else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")
    del engine


if __name__ == "__main__":
    main()
