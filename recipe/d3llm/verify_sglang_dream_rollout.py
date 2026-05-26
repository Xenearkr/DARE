#!/usr/bin/env python3
"""Phase 3B: HF (verl multiblock) vs SGLang FullAttnMultiBlock alignment verify.

Parallel usage (4 GPUs):
  bash recipe/d3llm/run_verify_sglang_parallel.sh

Single shard:
  CUDA_VISIBLE_DEVICES=0 python recipe/d3llm/verify_sglang_dream_rollout.py \\
    --backend hf --shard-id 0 --num-shards 4 --output-json /tmp/hf0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

DARE_ROOT = Path(__file__).resolve().parents[2]
SGLANG_PYTHON = DARE_ROOT / "third_party" / "sglang" / "python"
TRAIN_PARQUET = DARE_ROOT / "data/preprocessed/rl/train/lcbv5-K8_1.parquet"
DEFAULT_MODEL = DARE_ROOT / "models/finetune_d3LLM"
MASK_TOKEN_ID = 151666
PAD_TOKEN_ID = 151643

if str(DARE_ROOT) not in sys.path:
    sys.path.insert(0, str(DARE_ROOT))


@dataclass
class TaskResult:
    task_index: int
    backend: str
    prompt_tokens: int
    response_tokens: int
    nfe: int | None
    elapsed_s: float
    output_ids: list[int]
    text: str
    mask_in_output: bool


def parse_args():
    p = argparse.ArgumentParser(description="HF vs SGLang Dream multiblock alignment (phase 3B)")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--backend", choices=("hf", "sglang"), default=None)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--num-tasks", type=int, default=8, help="Total tasks across all shards")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--cache-delay-iter", type=int, default=32)
    p.add_argument("--mem-fraction-static", type=float, default=0.45)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument(
        "--merge-json",
        nargs=2,
        metavar=("HF_GLOB_OR_FILE", "SGLANG_GLOB_OR_FILE"),
        default=None,
        help="Merge HF and SGLang JSON shards and print alignment summary",
    )
    return p.parse_args()


def load_tasks(tokenizer, num_tasks: int, shard_id: int, num_shards: int) -> list[tuple[int, list[int]]]:
    import pandas as pd

    if not TRAIN_PARQUET.is_file():
        raise FileNotFoundError(TRAIN_PARQUET)
    df = pd.read_parquet(TRAIN_PARQUET)
    n = min(num_tasks, len(df))
    tasks: list[tuple[int, list[int]]] = []
    for i in range(n):
        if i % num_shards != shard_id:
            continue
        row = df.iloc[i]
        messages = row.get("prompt")
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        if isinstance(messages, str):
            messages = json.loads(messages)
        if hasattr(tokenizer, "apply_chat_template") and messages:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = str(messages)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 512:
            ids = ids[-512:]
        tasks.append((i, ids))
    return tasks


def left_pad_batch(prompt_ids: list[int], width: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    plen = len(prompt_ids)
    if plen > width:
        prompt_ids = prompt_ids[-width:]
        plen = width
    pad_left = width - plen
    ids = [pad_id] * pad_left + prompt_ids
    mask = [0] * pad_left + [1] * plen
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.long),
    )


def run_hf_task(model, tokenizer, task_index: int, prompt_ids: list[int], args) -> TaskResult:
    from verl.workers.rollout.dream_multiblock import execute_dream_multiblock_generation
    from verl.workers.rollout.rollout_utils import process_fastdream_generation_outputs

    idx, attn = left_pad_batch(prompt_ids, args.max_prompt_length, PAD_TOKEN_ID)
    device = next(model.parameters()).device
    idx = idx.to(device)
    attn = attn.to(device)
    gen_kwargs = {
        "gen_length": args.max_new_tokens,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "threshold": args.threshold,
        "block_add_threshold": 0.1,
        "decoded_token_threshold": 0.95,
        "cache_delay_iter": args.cache_delay_iter,
        "early_stop": True,
        "do_sample": args.temperature > 0,
        "per_sample_seed": False,
        "pad_token_id": PAD_TOKEN_ID,
        "dllm_decode": "multiblock",
    }
    t0 = time.time()
    responses, _, _, _ = execute_dream_multiblock_generation(
        module=model,
        gen_kwargs=gen_kwargs,
        idx_repeat=idx,
        attention_mask_repeat=attn,
        response_length=args.max_new_tokens,
        tokenizer=tokenizer,
        process_outputs_fn=process_fastdream_generation_outputs,
    )
    elapsed = time.time() - t0
    resp_ids = responses[0].tolist()
    text = tokenizer.decode(resp_ids, skip_special_tokens=True)
    return TaskResult(
        task_index=task_index,
        backend="hf",
        prompt_tokens=len(prompt_ids),
        response_tokens=len(resp_ids),
        nfe=None,
        elapsed_s=elapsed,
        output_ids=resp_ids,
        text=text,
        mask_in_output=any(t == MASK_TOKEN_ID for t in resp_ids),
    )


def build_sglang_engine(model_path: Path, args):
    if str(SGLANG_PYTHON) not in sys.path:
        sys.path.insert(0, str(SGLANG_PYTHON))
    import yaml
    from sglang.srt.entrypoints.engine import Engine

    algo_cfg = {
        "threshold": args.threshold,
        "block_add_threshold": 0.1,
        "decoded_token_threshold": 0.95,
        "block_size": args.block_length,
        "temperature": args.temperature,
        "top_p": 1.0 if args.temperature <= 0 else 0.95,
        "cache_delay_iter": args.cache_delay_iter,
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
        mem_fraction_static=args.mem_fraction_static,
        disable_cuda_graph=True,
        attention_backend="torch_native",
        max_running_requests=1,
        dllm_algorithm="FullAttnMultiBlock",
        dllm_algorithm_config=algo_path,
    )


def run_sglang_task(engine, tokenizer, task_index: int, prompt_ids: list[int], args) -> TaskResult:
    t0 = time.time()
    out = engine.generate(
        prompt=None,
        input_ids=[prompt_ids],
        sampling_params={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": 1.0 if args.temperature <= 0 else 0.95,
            "sampling_seed": 42 + task_index,
        },
        return_logprob=False,
    )
    elapsed = time.time() - t0
    if isinstance(out, list):
        out = out[0] if out else {}
    output_ids = out.get("output_ids") or []
    meta = out.get("meta_info") or {}
    nfe_raw = meta.get("nfe")
    nfe = nfe_raw[0] if isinstance(nfe_raw, list) and nfe_raw else nfe_raw
    text = out.get("text") or tokenizer.decode(output_ids, skip_special_tokens=True)
    return TaskResult(
        task_index=task_index,
        backend="sglang",
        prompt_tokens=len(prompt_ids),
        response_tokens=len(output_ids),
        nfe=int(nfe) if nfe is not None else None,
        elapsed_s=elapsed,
        output_ids=output_ids,
        text=text,
        mask_in_output=any(t == MASK_TOKEN_ID for t in output_ids),
    )


def load_hf_model(model_path: Path):
    from transformers import AutoModel, AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(model_path), trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    return model, tokenizer


def run_backend(args):
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    tasks = load_tasks(tokenizer, args.num_tasks, args.shard_id, args.num_shards)
    print(f"[{args.backend}] shard {args.shard_id}/{args.num_shards} tasks={len(tasks)}")

    results: list[dict[str, Any]] = []
    if args.backend == "hf":
        model, _ = load_hf_model(args.model_path)
        for task_index, prompt_ids in tasks:
            r = run_hf_task(model, tokenizer, task_index, prompt_ids, args)
            print(f"  task={task_index} plen={r.prompt_tokens} rtok={r.response_tokens} "
                  f"elapsed={r.elapsed_s:.1f}s mask={r.mask_in_output}")
            results.append(asdict(r))
        del model
        torch.cuda.empty_cache()
    else:
        engine = build_sglang_engine(args.model_path, args)
        for task_index, prompt_ids in tasks:
            r = run_sglang_task(engine, tokenizer, task_index, prompt_ids, args)
            print(f"  task={task_index} plen={r.prompt_tokens} rtok={r.response_tokens} "
                  f"nfe={r.nfe} elapsed={r.elapsed_s:.1f}s mask={r.mask_in_output}")
            results.append(asdict(r))

    out_path = args.output_json or Path(f"/tmp/dream_verify_{args.backend}_s{args.shard_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"backend": args.backend, "shard_id": args.shard_id, "results": results}, f, indent=2)
    print(f"[{args.backend}] wrote {out_path}")
    return out_path


def merge_results(hf_paths: list[Path], sg_paths: list[Path]):
    def load_all(paths: list[Path]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for p in paths:
            with open(p) as f:
                data = json.load(f)
            for row in data.get("results", []):
                out[row["task_index"]] = row
        return out

    hf = load_all(hf_paths)
    sg = load_all(sg_paths)
    common = sorted(set(hf) & set(sg))
    if not common:
        raise SystemExit("[FAIL] no overlapping task indices between HF and SGLang JSON")

    token_match = text_match = 0
    nfe_pairs: list[tuple[int, int | None, int | None]] = []
    print(f"\n{'idx':>4} {'plen':>5} {'hf_tok':>7} {'sg_tok':>7} {'tok_eq':>7} {'text_eq':>8} hf_s sg_s")
    for i in common:
        h, s = hf[i], sg[i]
        te = h["output_ids"] == s["output_ids"]
        tx = h["text"].strip() == s["text"].strip()
        token_match += int(te)
        text_match += int(tx)
        nfe_pairs.append((i, h.get("nfe"), s.get("nfe")))
        print(f"{i:4d} {h['prompt_tokens']:5d} {h['response_tokens']:7d} {s['response_tokens']:7d} "
              f"{str(te):>7} {str(tx):>8} {h['elapsed_s']:4.1f} {s['elapsed_s']:4.1f}")

    n = len(common)
    hf_time = sum(hf[i]["elapsed_s"] for i in common)
    sg_time = sum(sg[i]["elapsed_s"] for i in common)
    print(f"\nSUMMARY tasks={n} token_match={token_match}/{n} ({100*token_match/n:.1f}%) "
          f"text_match={text_match}/{n} ({100*text_match/n:.1f}%)")
    print(f"SUMMARY total_time hf={hf_time:.1f}s sglang={sg_time:.1f}s speedup={hf_time/max(sg_time,1e-6):.2f}x")
    if token_match == n and not any(hf[i]["mask_in_output"] or sg[i]["mask_in_output"] for i in common):
        print("[PASS] phase 3B alignment verify")
        return 0
    print("[WARN] outputs differ — review per-task (temperature=0 expected for strict match)")
    return 0 if text_match >= max(1, n // 2) else 1


def _expand_glob(pattern: str) -> list[Path]:
    from glob import glob

    paths = sorted(Path(p) for p in glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files match: {pattern}")
    return paths


def main():
    args = parse_args()
    if args.merge_json:
        hf_paths = _expand_glob(args.merge_json[0])
        sg_paths = _expand_glob(args.merge_json[1])
        raise SystemExit(merge_results(hf_paths, sg_paths))
    if args.backend is None:
        raise SystemExit("Specify --backend hf|sglang or --merge-json")
    run_backend(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
