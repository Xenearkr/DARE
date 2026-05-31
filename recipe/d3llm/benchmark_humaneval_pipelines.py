#!/usr/bin/env python3
"""HumanEval: HF multiblock vs SGLang train/val pipelines (4-GPU, no training code changes).

OOM-safe: HF and SGLang run in separate process batches (one model per GPU at a time).

Example:
  conda activate DARE
  export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
  python recipe/d3llm/benchmark_humaneval_pipelines.py \\
    --limit 82 --ngpus 4 \\
    --output-json logs/benchmarks/humaneval_pipelines_latest.json
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

DARE_ROOT = Path(__file__).resolve().parents[2]
RECIPE_D3LLM = Path(__file__).resolve().parent
DEFAULT_PARQUET = DARE_ROOT / "data/preprocessed/rl/test/humaneval_1.parquet"
DEFAULT_MODEL = DARE_ROOT / "models/finetune_d3LLM"
DEFAULT_OUT = DARE_ROOT / "logs/benchmarks/humaneval_pipelines_latest.json"

if str(DARE_ROOT) not in sys.path:
    sys.path.insert(0, str(DARE_ROOT))
if str(RECIPE_D3LLM) not in sys.path:
    sys.path.insert(0, str(RECIPE_D3LLM))

from compare_sglang_train_vs_val_path import (  # noqa: E402
    load_prompt_ids,
    run_train_path,
    run_val_path,
)
from compare_sglang_train_vs_val_path import SGLANG_PYTHON  # noqa: E402

if str(SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_PYTHON))

PATHS = ("hf_val", "hf_train", "sglang_val", "sglang_train")
MASK_TOKEN_ID = 151666


def resolve_hf_path_specs(hf_paths: str, train_temperature: float) -> list[tuple[str, float]]:
    """Map --hf-paths to (path_name, temperature) pairs."""
    if hf_paths == "both":
        return [("hf_val", 0.0), ("hf_train", train_temperature)]
    if hf_paths == "train":
        return [("hf_train", train_temperature)]
    if hf_paths == "val":
        return [("hf_val", 0.0)]
    raise ValueError(f"Unknown hf_paths={hf_paths!r}")


def resolve_hf_path_names(hf_paths: str) -> tuple[str, ...]:
    return tuple(name for name, _ in resolve_hf_path_specs(hf_paths, train_temperature=0.0))


def _count_response_tokens(resp_ids: list[int], pad_token_id: int) -> int:
    """Non-pad response tokens (excludes trailing padding and unrevealed masks)."""
    return sum(1 for tid in resp_ids if tid not in (pad_token_id, MASK_TOKEN_ID))


def _tpf(gen_tokens: int, nfe: int) -> float:
    """Tokens per forward pass (diffusion efficiency metric)."""
    return round(gen_tokens / nfe, 4) if nfe > 0 else 0.0


@dataclass
class WorkerArgs:
    gpu_rank: int
    row_indices: list[int]
    model_path: str
    parquet: str
    max_prompt_length: int
    max_new_tokens: int
    train_temperature: float
    val_temperature: float
    threshold: float
    mem_fraction_static: float
    train_seed: int
    shard_path: str
    hf_shard_path: str = ""
    only_sglang_val: bool = False
    hf_paths: str = "both"


def build_training_aligned_engine(model_path: Path, max_new_tokens: int, threshold: float, mem_fraction: float, temperature: float = 0.0):
    """Mirror sglang_dream_rollout._init_inference_engine (single-GPU smoke)."""
    from sglang.srt.entrypoints.engine import Engine

    max_prompt = 1024
    max_seq_tokens = max_prompt + max_new_tokens + 64
    max_prefill_tokens = max(4096, max_seq_tokens)
    algo_cfg = {
        "threshold": threshold,
        "block_add_threshold": 0.1,
        "decoded_token_threshold": 0.95,
        "block_size": 32,
        "temperature": temperature,
        "top_p": 1.0,
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
        enable_memory_saver=True,
        disable_cuda_graph=True,
        attention_backend="torch_native",
        max_running_requests=1,
        max_prefill_tokens=max_prefill_tokens,
        chunked_prefill_size=-1,
        dllm_algorithm="FullAttnMultiBlock",
        dllm_algorithm_config=algo_path,
    )


class _BenchArgs:
    def __init__(self, wa: WorkerArgs):
        self.train_temperature = wa.train_temperature
        self.val_temperature = wa.val_temperature
        self.max_new_tokens = wa.max_new_tokens
        self.train_seed = wa.train_seed


def load_hf_model(model_path: Path, device: torch.device):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return model.to(device).eval()


def run_hf_row(
    model,
    tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    pad_token_id: int,
) -> dict[str, Any]:
    from verl.workers.rollout.dream_multiblock import (
        DreamMultiBlockConfig,
        _estimate_max_nfe,
        _run_multiblock,
        _strip_left_padding,
        bind_multiblock,
    )

    device = next(model.parameters()).device
    mb_cfg = DreamMultiBlockConfig(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        threshold=0.5,
        block_add_threshold=0.1,
        decoded_token_threshold=0.95,
        cache_delay_iter=32,
        early_stop=True,
    )
    bind_multiblock(model, cfg=mb_cfg)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    compact_ids, compact_mask, prompt_lengths = _strip_left_padding(input_ids, attention_mask, pad_token_id)
    plen = int(prompt_lengths[0].item())
    max_nfe = _estimate_max_nfe(max_new_tokens, mb_cfg.block_length)
    top_p = 0.95 if temperature > 0 else 1.0
    out, nfe = _run_multiblock(
        model,
        compact_ids,
        compact_mask,
        max_new_tokens,
        mb_cfg,
        mb_cfg.block_length,
        temperature,
        top_p,
        max_nfe,
    )
    resp_ids = out[0, plen:].tolist()
    gen_tokens = _count_response_tokens(resp_ids, pad_token_id)
    text = tokenizer.decode(resp_ids, skip_special_tokens=True)
    return {
        "text": text,
        "nfe": nfe,
        "gen_tokens": gen_tokens,
        "tpf": _tpf(gen_tokens, nfe),
        "temperature": temperature,
    }


def score_humaneval(ground_truth: str, text: str) -> dict[str, Any]:
    from verl.utils.reward_score.code_reward import rllm_reward_fn_code

    out = rllm_reward_fn_code("humaneval", text, ground_truth, {})
    return {
        "pass": bool(out.get("is_correct")),
        "reward": float(out.get("reward", 0.0)),
        "pred": out.get("pred") or "",
    }


def _compact_entry_paths(entry: dict[str, Any]) -> None:
    for p in PATHS:
        if p not in entry:
            continue
        blob = entry[p]
        if "text" in blob:
            t = blob["text"]
            blob["text_head"] = t[:240]
            blob["text_len"] = len(t)
            del blob["text"]
        if "raw_text" in blob:
            rt = blob["raw_text"]
            blob["raw_text_head"] = rt[:240]
            blob["raw_len"] = len(rt)
            del blob["raw_text"]


def hf_gpu_worker(wa: WorkerArgs) -> None:
    """Phase 1: HF only; exit process frees GPU before SGLang."""
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    device = torch.device(f"cuda:{wa.gpu_rank}")
    torch.cuda.set_device(device)

    from transformers import AutoTokenizer

    model_path = Path(wa.model_path)
    parquet = Path(wa.parquet)
    df = pd.read_parquet(parquet)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643

    rows_out: list[dict[str, Any]] = []
    t0 = time.time()
    path_specs = resolve_hf_path_specs(wa.hf_paths, wa.train_temperature)
    primary_path = path_specs[0][0]
    hf_model = load_hf_model(model_path, device)
    try:
        for i, row in enumerate(wa.row_indices):
            gt = df.iloc[row]["reward_model"]["ground_truth"]
            prompt_ids, _ = load_prompt_ids(tokenizer, parquet, row, wa.max_prompt_length)
            entry: dict[str, Any] = {"row": row}
            for path_name, temp in path_specs:
                t_row = time.time()
                gen = run_hf_row(hf_model, tokenizer, prompt_ids, wa.max_new_tokens, temp, pad_id)
                entry[path_name] = {**gen, **score_humaneval(gt, gen["text"])}
                print(
                    f"[gpu{wa.gpu_rank}] row={row} {path_name} "
                    f"pass={entry[path_name]['pass']} nfe={gen['nfe']} "
                    f"gen_tokens={gen['gen_tokens']} tpf={gen['tpf']:.4f} "
                    f"elapsed={time.time() - t_row:.1f}s",
                    flush=True,
                )
            rows_out.append(entry)
            if (i + 1) % 5 == 0 or i + 1 == len(wa.row_indices):
                n_pass = sum(1 for e in rows_out if e.get(primary_path, {}).get("pass"))
                avg_tpf = sum(e.get(primary_path, {}).get("tpf", 0) for e in rows_out) / len(rows_out)
                print(
                    f"[gpu{wa.gpu_rank}] HF {i + 1}/{len(wa.row_indices)} "
                    f"{primary_path}_pass={n_pass}/{len(rows_out)} avg_tpf={avg_tpf:.4f} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    finally:
        del hf_model
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    for entry in rows_out:
        _compact_entry_paths(entry)
    Path(wa.shard_path).parent.mkdir(parents=True, exist_ok=True)
    with open(wa.shard_path, "w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2)
    print(f"[gpu{wa.gpu_rank}] HF shard {wa.shard_path}", flush=True)


def sglang_gpu_worker(wa: WorkerArgs) -> None:
    """Phase 2: SGLang only (fresh process, full GPU for Engine)."""
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    device = torch.device(f"cuda:{wa.gpu_rank}")
    torch.cuda.set_device(device)

    from transformers import AutoTokenizer

    shard_dir = Path(wa.hf_shard_path).parent
    rows_out: list[dict[str, Any]] = []
    for p in sorted(shard_dir.glob("rank*_hf.json")):
        rows_out.extend(json.loads(p.read_text(encoding="utf-8")))
    by_row = {e["row"]: e for e in rows_out}
    rows_out = [by_row[row] for row in wa.row_indices]

    model_path = Path(wa.model_path)
    parquet = Path(wa.parquet)
    df = pd.read_parquet(parquet)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    bench_args = _BenchArgs(wa)

    t0 = time.time()
    engine = build_training_aligned_engine(
        model_path, wa.max_new_tokens, wa.threshold, wa.mem_fraction_static, temperature=0.0
    )
    try:
        for i, row in enumerate(wa.row_indices):
            entry = by_row[row]
            gt = df.iloc[row]["reward_model"]["ground_truth"]
            prompt_ids, _ = load_prompt_ids(tokenizer, parquet, row, wa.max_prompt_length)
            val_r = run_val_path(engine, tokenizer, prompt_ids, bench_args)
            paths_to_run = [("sglang_val", val_r)]
            if not wa.only_sglang_val:
                train_r = run_train_path(engine, tokenizer, prompt_ids, bench_args)
                paths_to_run.append(("sglang_train", train_r))
            for path_name, run_r in paths_to_run:
                text = run_r["finalized_text"]
                entry[path_name] = {
                    "text": text,
                    "raw_text": run_r["raw_text"],
                    "nfe": run_r.get("nfe"),
                    "gen_tokens": run_r.get("gen_tokens"),
                    "tpf": run_r.get("tpf"),
                    "finish_reason": run_r.get("finish_reason"),
                    "text_eq_raw": text == run_r["raw_text"],
                    **score_humaneval(gt, text),
                }
            if wa.only_sglang_val:
                entry.pop("sglang_train", None)
            if (i + 1) % 5 == 0 or i + 1 == len(wa.row_indices):
                n_pass = sum(1 for e in rows_out if e.get("sglang_val", {}).get("pass"))
                print(
                    f"[gpu{wa.gpu_rank}] SGLang {i + 1}/{len(wa.row_indices)} "
                    f"sglang_val_pass={n_pass}/{len(rows_out)} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    for entry in rows_out:
        _compact_entry_paths(entry)
    with open(wa.shard_path, "w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2)
    print(f"[gpu{wa.gpu_rank}] merged shard {wa.shard_path}", flush=True)


def run_phase_inprocess(worker_fn, worker_args: list[WorkerArgs], phase_name: str) -> None:
    """Run workers in the parent process (no mp.spawn); one GPU job at a time."""
    print(f"\n=== Phase: {phase_name} (in-process, {len(worker_args)} jobs) ===", flush=True)
    for wa in worker_args:
        worker_fn(wa)


def run_phase(worker_fn, worker_args: list[WorkerArgs], phase_name: str, serial: bool = False) -> None:
    print(f"\n=== Phase: {phase_name} ({len(worker_args)} workers, serial={serial}) ===", flush=True)
    if serial:
        for wa in worker_args:
            ctx = mp.get_context("spawn")
            p = ctx.Process(target=worker_fn, args=(wa,))
            p.start()
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"{phase_name} worker failed with exit code {p.exitcode}")
            time.sleep(1)
        return
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker_fn, args=(wa,)) for wa in worker_args]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(f"{phase_name} worker failed with exit code {p.exitcode}")
    time.sleep(2)


def _tpf_summary(rows: list[dict[str, Any]], path_name: str) -> dict[str, float]:
    from verl.utils.reward_score.code_efficiency import normalize_rollout_nfe

    tpfs = [r[path_name]["tpf"] for r in rows if path_name in r and "tpf" in r[path_name]]
    nfes = [
        normalize_rollout_nfe(r[path_name]["nfe"])
        for r in rows
        if path_name in r and "nfe" in r[path_name]
    ]
    if not tpfs:
        return {}
    return {
        "mean": round(sum(tpfs) / len(tpfs), 4),
        "min": round(min(tpfs), 4),
        "max": round(max(tpfs), 4),
        "mean_nfe": round(sum(nfes) / len(nfes), 2),
    }


def analyze(rows: list[dict[str, Any]], active_paths: tuple[str, ...] | None = None) -> dict[str, Any]:
    paths = active_paths or PATHS
    n = len(rows)
    pass_rates = {p: sum(1 for r in rows if r.get(p, {}).get("pass")) / n if n else 0.0 for p in paths}
    tpf_stats = {p: _tpf_summary(rows, p) for p in paths if p.startswith(("hf_", "sglang_"))}

    def cross(a: str, b: str) -> dict[str, int]:
        both = a_only = b_only = neither = 0
        for r in rows:
            pa = r.get(a, {}).get("pass", False)
            pb = r.get(b, {}).get("pass", False)
            if pa and pb:
                both += 1
            elif pa:
                a_only += 1
            elif pb:
                b_only += 1
            else:
                neither += 1
        return {"both_pass": both, f"{a}_only": a_only, f"{b}_only": b_only, "both_fail": neither}

    hf_ok_sgl_val_fail = [
        r["row"] for r in rows if r.get("hf_val", {}).get("pass") and not r.get("sglang_val", {}).get("pass")
    ]
    sgl_train_val_diff = sum(
        1 for r in rows if r.get("sglang_train", {}).get("pass") != r.get("sglang_val", {}).get("pass")
    )
    sgl_raw_vs_fin = sum(1 for r in rows if r.get("sglang_val", {}).get("text_eq_raw") is False)

    fail_reasons = Counter()
    for r in rows:
        if r.get("sglang_val", {}).get("pass"):
            continue
        if not ((r.get("sglang_val") or {}).get("pred") or "").strip():
            fail_reasons["empty_pred"] += 1
        elif r.get("hf_val", {}).get("pass"):
            fail_reasons["hf_pass_sgl_fail"] += 1
        else:
            fail_reasons["both_fail"] += 1

    verdict = []
    gap = pass_rates.get("hf_val", 0) - pass_rates.get("sglang_val", 0)
    if "sglang_val" in pass_rates and gap > 0.15:
        verdict.append(
            f"主因：SGLang val 比 HF multiblock 低 {gap:.1%} "
            f"({pass_rates['hf_val']:.1%} vs {pass_rates['sglang_val']:.1%})，问题在 SGLang 解码/后处理。"
        )
    if sgl_train_val_diff > n * 0.05:
        verdict.append(f"次要：sglang train vs val 不一致 {sgl_train_val_diff}/{n} 题。")
    if sgl_raw_vs_fin > 0:
        verdict.append(f"finalize 改变输出：{sgl_raw_vs_fin}/{n} 题。")

    return {
        "n": n,
        "pass_at_1": pass_rates,
        "tpf": tpf_stats,
        "cross_hf_val_vs_sglang_val": cross("hf_val", "sglang_val"),
        "cross_hf_val_vs_sglang_train": cross("hf_val", "sglang_train"),
        "cross_sglang_train_vs_val": cross("sglang_train", "sglang_val"),
        "hf_val_pass_sglang_val_fail_rows": hf_ok_sgl_val_fail[:30],
        "hf_val_pass_sglang_val_fail_count": len(hf_ok_sgl_val_fail),
        "sglang_val_fail_reasons": dict(fail_reasons),
        "verdict_zh": verdict,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--limit", type=int, default=82)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--ngpus", type=int, default=4)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--train-temperature", type=float, default=0.2)
    p.add_argument("--val-temperature", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.55,
        help="SGLang-only benchmark: ~0.55 on empty GPU; training smoke uses 0.32 beside FSDP",
    )
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--only-hf",
        action="store_true",
        help="Run HF multiblock only (skip SGLang phase)",
    )
    p.add_argument(
        "--hf-paths",
        choices=("both", "train", "val"),
        default="both",
        help="HF decode paths to run: both (default), train (hf_train T=train_temperature only), val",
    )
    p.add_argument(
        "--only-sglang-val",
        action="store_true",
        help="Re-run aligned sglang_val only; requires existing rank*_hf.json shards",
    )
    p.add_argument(
        "--serial-sglang",
        action="store_true",
        help="Start one SGLang worker at a time (avoids parallel Engine OOM)",
    )
    p.add_argument(
        "--hf-shards-dir",
        type=Path,
        default=None,
        help="Directory with rank*_hf.json when using --only-sglang-val",
    )
    p.add_argument(
        "--inprocess",
        action="store_true",
        help="Run SGLang workers in parent process (no mp.spawn)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.parquet)
    end = min(args.offset + args.limit, len(df))
    all_rows = list(range(args.offset, end))
    ngpus = min(args.ngpus, len(all_rows), torch.cuda.device_count())
    if ngpus < 1:
        raise RuntimeError("No CUDA devices visible")

    shards_dir = args.hf_shards_dir or (args.output_json.parent / f"{args.output_json.stem}_shards")
    shards_dir.mkdir(parents=True, exist_ok=True)

    buckets: list[list[int]] = [[] for _ in range(ngpus)]
    for row in all_rows:
        buckets[row % ngpus].append(row)

    hf_workers: list[WorkerArgs] = []
    sgl_workers: list[WorkerArgs] = []
    for rank in range(ngpus):
        if not buckets[rank]:
            continue
        hf_shard = str(shards_dir / f"rank{rank}_hf.json")
        final_shard = str(shards_dir / f"rank{rank}.json")
        base = dict(
            gpu_rank=rank,
            row_indices=buckets[rank],
            model_path=str(args.model_path),
            parquet=str(args.parquet),
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            train_temperature=args.train_temperature,
            val_temperature=args.val_temperature,
            threshold=args.threshold,
            mem_fraction_static=args.mem_fraction_static,
            train_seed=args.train_seed,
            only_sglang_val=args.only_sglang_val,
            hf_paths=args.hf_paths,
        )
        if not args.only_sglang_val:
            hf_workers.append(WorkerArgs(**base, shard_path=hf_shard))
        elif not Path(hf_shard).is_file():
            raise FileNotFoundError(f"--only-sglang-val needs {hf_shard}")
        if not args.only_hf:
            sgl_workers.append(
                WorkerArgs(**base, shard_path=final_shard, hf_shard_path=hf_shard)
            )

    print(
        f"HumanEval benchmark: rows={len(all_rows)} ngpus={ngpus} "
        f"mem_fraction={args.mem_fraction_static} only_hf={args.only_hf} "
        f"hf_paths={args.hf_paths} only_sglang_val={args.only_sglang_val}",
        flush=True,
    )
    t0 = time.time()

    if not args.only_sglang_val:
        run_phase(hf_gpu_worker, hf_workers, "HF multiblock")
    if not args.only_hf:
        sgl_phase = "SGLang val (per-sample)" if args.only_sglang_val else "SGLang train/val"
        if args.inprocess:
            run_phase_inprocess(sglang_gpu_worker, sgl_workers, sgl_phase)
        else:
            run_phase(
                sglang_gpu_worker,
                sgl_workers,
                sgl_phase,
                serial=args.serial_sglang or args.only_sglang_val,
            )

    merged: list[dict[str, Any]] = []
    if args.only_hf:
        for wa in hf_workers:
            with open(wa.shard_path, encoding="utf-8") as f:
                merged.extend(json.load(f))
    else:
        for wa in sgl_workers:
            with open(wa.shard_path, encoding="utf-8") as f:
                merged.extend(json.load(f))
    merged.sort(key=lambda x: x["row"])

    hf_active = resolve_hf_path_names(args.hf_paths)
    if args.only_hf:
        active_paths = hf_active
    elif args.only_sglang_val:
        active_paths = hf_active + ("sglang_val",)
    else:
        active_paths = hf_active + ("sglang_val", "sglang_train")
    analysis = analyze(merged, active_paths=active_paths)
    report = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "paths": list(active_paths),
        "sglang_val_per_sample_aligned": not args.only_hf,
        "elapsed_s": round(time.time() - t0, 2),
        "analysis": analysis,
        "details": merged,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {args.output_json} (elapsed={report['elapsed_s']}s)")
    print("\n--- pass@1 ---")
    for k, v in analysis["pass_at_1"].items():
        print(f"  {k}: {v:.1%}")
    if analysis.get("tpf"):
        print("\n--- TPF (tokens per forward) ---")
        for k, v in analysis["tpf"].items():
            if v:
                print(f"  {k}: mean={v['mean']:.4f} nfe_mean={v['mean_nfe']:.1f} (min={v['min']}, max={v['max']})")
    if not args.only_hf:
        print("\n--- hf_val vs sglang_val ---")
        print(json.dumps(analysis["cross_hf_val_vs_sglang_val"], indent=2))
        print(f"\nhf_val pass & sglang_val fail: {analysis['hf_val_pass_sglang_val_fail_count']}")
        print("\n--- 结论 ---")
        for line in analysis["verdict_zh"]:
            print(f"  • {line}")


if __name__ == "__main__":
    main()
