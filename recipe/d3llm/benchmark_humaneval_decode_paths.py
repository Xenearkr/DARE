#!/usr/bin/env python3
"""HumanEval pass@1: HF multiblock vs SGLang train/val paths (4-GPU data parallel).

Each GPU runs one worker (same layout as training: ``tensor_model_parallel_size=1``,
one SGLang Engine per rank). Engine kwargs match ``SGLangDreamRollout`` / smoke script.

Example (use DARE conda env):
  export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
  python recipe/d3llm/benchmark_humaneval_decode_paths.py \\
    --limit 164 --paths hf_multiblock,sglang_train,sglang_val \\
    --output-json logs/benchmarks/humaneval_decode_paths.json
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import gc
import importlib.util
import json
import os
import re
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import subprocess

DARE_ROOT = Path(__file__).resolve().parents[2]
RECIPE_D3LLM = Path(__file__).resolve().parent
SGLANG_PYTHON = DARE_ROOT / "third_party" / "sglang" / "python"
DEFAULT_MODEL = DARE_ROOT / "models" / "finetune_d3LLM"
DEFAULT_PARQUET = DARE_ROOT / "data/preprocessed/rl/test/humaneval_1.parquet"
MASK_TOKEN_ID = 151666
PAD_TOKEN_ID = 151643

if str(DARE_ROOT) not in sys.path:
    sys.path.insert(0, str(DARE_ROOT))
if str(RECIPE_D3LLM) not in sys.path:
    sys.path.insert(0, str(RECIPE_D3LLM))
if str(SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_PYTHON))


def parse_args():
    p = argparse.ArgumentParser(description="HumanEval pass@1: HF vs SGLang (4-GPU DP)")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--limit", type=int, default=164)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument(
        "--paths",
        type=str,
        default="hf_multiblock,sglang_train,sglang_val",
        help="hf_multiblock,sglang_train,sglang_val",
    )
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--train-temperature", type=float, default=0.2)
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help="Default: 0.32 (smoke training colocated). Use 0.45 for SGLang-only GPU.",
    )
    p.add_argument(
        "--smoke-mem",
        action="store_true",
        help="Force mem_fraction_static=0.32 (training smoke SGLang)",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument(
        "--merge-json",
        type=Path,
        default=None,
        help="Skip paths already present in this JSON",
    )
    p.add_argument("--partial-dir", type=Path, default=None, help="Per-rank partial JSON dir")
    p.add_argument("--worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _fix_worker_env(env: dict) -> dict:
    env = dict(env)
    env.setdefault("HF_HUB_OFFLINE", "1")
    # Triton looks up libcuda via `/sbin/ldconfig`; on Ubuntu it lives under /usr/sbin.
    path = env.get("PATH", "")
    for prefix in ("/usr/sbin", "/sbin"):
        if prefix not in path.split(":"):
            path = f"{prefix}:{path}"
    env["PATH"] = path
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return env


def _launch_workers(args, args_dict: dict, paths_to_run: List[str], partial_dir: Path) -> None:
    """One OS process per GPU (like Ray workers), not torch mp.spawn."""
    this = Path(__file__).resolve()
    python = sys.executable
    procs: List[subprocess.Popen] = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    visible = [x.strip() for x in visible if x.strip()]
    for rank in range(args.world_size):
        if rank >= len(visible):
            raise RuntimeError(f"rank {rank} >= len(CUDA_VISIBLE_DEVICES)={visible}")
        env = _fix_worker_env(os.environ.copy())
        env["CUDA_VISIBLE_DEVICES"] = visible[rank]
        cmd = [
            python,
            str(this),
            "--worker-rank",
            str(rank),
            "--world-size",
            str(args.world_size),
            "--model-path",
            str(args.model_path),
            "--parquet",
            str(args.parquet),
            "--limit",
            str(args.limit),
            "--offset",
            str(args.offset),
            "--paths",
            ",".join(paths_to_run),
            "--max-prompt-length",
            str(args.max_prompt_length),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--train-temperature",
            str(args.train_temperature),
            "--train-seed",
            str(args.train_seed),
            "--partial-dir",
            str(partial_dir),
        ]
        if args.mem_fraction_static is not None:
            cmd.extend(["--mem-fraction-static", str(args.mem_fraction_static)])
        if args.smoke_mem:
            cmd.append("--smoke-mem")
        print(f"[launcher] rank {rank}: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(DARE_ROOT)))
        time.sleep(5)
    failed = []
    for rank, proc in enumerate(procs):
        code = proc.wait()
        if code != 0:
            failed.append((rank, code))
    if failed:
        raise RuntimeError(f"Worker(s) failed: {failed}")


def prompt_to_ids(tokenizer, messages, max_prompt_length: int) -> List[int]:
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
    return ids


def _extract_code_from_model(model_response: str) -> Optional[str]:
    code_blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    if len(code_blocks) == 1:
        return code_blocks[0].strip()
    for block in code_blocks:
        block = block.strip()
        if re.search(r"def\s+\w+\s*\(", block):
            return block
    return code_blocks[-1].strip()


def _humanevalplus_run_test():
    name = "verl.utils.reward_score.code_utils.humanevalplus"
    if name in sys.modules:
        return sys.modules[name].run_test
    rs_root = DARE_ROOT / "verl" / "utils" / "reward_score"
    if "verl.utils.reward_score.code_utils" not in sys.modules:
        pkg = types.ModuleType("verl.utils.reward_score.code_utils")
        pkg.__path__ = [str(rs_root / "code_utils")]
        sys.modules["verl.utils.reward_score.code_utils"] = pkg
    spec = importlib.util.spec_from_file_location(name, rs_root / "code_utils" / "humanevalplus.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.run_test


def _humaneval_pass(model_response: str, test: str) -> bool:
    code = _extract_code_from_model(model_response)
    if not code:
        return False
    num_test_cases = 0
    try:
        parsed = ast.parse(test)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Assert):
                num_test_cases += 1
    except SyntaxError:
        num_test_cases = test.count("assert ")
    func_match = re.search(r"def\s+(\w+)\s*\(", code)
    if func_match:
        test = test + f"\n\n# Execute the tests\ncheck({func_match.group(1)})"
    run_test = _humanevalplus_run_test()
    succ, _ = run_test(code, test, max(1, num_test_cases))
    return bool(succ)


def score_row(row: dict, text: str) -> bool:
    rm = row["reward_model"]
    gt = rm.get("ground_truth") if isinstance(rm, dict) else rm
    return _humaneval_pass(text, gt)


def _mem_fraction(args) -> float:
    if args.mem_fraction_static is not None:
        return float(args.mem_fraction_static)
    if args.smoke_mem:
        return 0.32
    return 0.45


async def _sglang_generate_async(engine, tokenizer, prompt_ids: List[int], sampling: dict, max_new_tokens: int):
    from verl.workers.rollout.dream_rollout_debug import format_nfe_for_log
    from verl.workers.rollout.sglang_rollout.sglang_dream_rollout import (
        _dream_stop_token_ids,
        _finalize_dream_response_tensor,
    )
    from verl.workers.rollout.sglang_rollout.sglang_rollout import _post_process_outputs

    stop_ids = _dream_stop_token_ids(tokenizer, PAD_TOKEN_ID, PAD_TOKEN_ID, MASK_TOKEN_ID)
    params = dict(sampling)
    params.setdefault("stop_token_ids", sorted(stop_ids))
    params.setdefault("skip_special_tokens", True)
    params.setdefault("spaces_between_special_tokens", True)
    out = await engine.async_generate(
        prompt=None,
        sampling_params=params,
        return_logprob=True,
        input_ids=[prompt_ids],
        image_data=None,
    )
    if isinstance(out, list):
        out = out[0]
    raw_resp, _ = _post_process_outputs(tokenizer, [out])
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    fr = meta.get("finish_reason")
    opening = tokenizer.encode("To solve this problem", add_special_tokens=False)
    finalized, _ = _finalize_dream_response_tensor(
        raw_resp,
        None,
        stop_ids,
        PAD_TOKEN_ID,
        max_new_tokens,
        finish_reasons=[fr],
        opening_prefix_ids=opening,
    )
    text = tokenizer.decode(finalized[0].tolist(), skip_special_tokens=True)
    return text, format_nfe_for_log(meta.get("nfe"))


def sglang_generate_text(engine, tokenizer, prompt_ids: List[int], sampling: dict, max_new_tokens: int):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _sglang_generate_async(engine, tokenizer, prompt_ids, sampling, max_new_tokens)
        )
    finally:
        loop.close()


def hf_multiblock_generate(
    model,
    tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    seed: Optional[int],
) -> tuple[str, str]:
    from verl.workers.rollout.dream_multiblock import DreamMultiBlockConfig, bind_multiblock

    mb_cfg = DreamMultiBlockConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        block_length=32,
        threshold=0.5,
        block_add_threshold=0.1,
        decoded_token_threshold=0.95,
        cache_delay_iter=32,
        early_stop=True,
    )
    bind_multiblock(model, cfg=mb_cfg)
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    attn = torch.ones_like(input_ids)
    top_p = 0.95 if temperature > 0 else 1.0
    if seed is not None and temperature > 0:
        torch.manual_seed(seed)
    with torch.no_grad():
        out, nfe = model.diffusion_generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            threshold=mb_cfg.threshold,
            block_length=mb_cfg.block_length,
            return_dict_in_generate=True,
        )
    if hasattr(out, "sequences"):
        seq = out.sequences[0]
    else:
        seq = out[0] if isinstance(out, (tuple, list)) else out
    plen = len(prompt_ids)
    if len(seq) > plen:
        text = tokenizer.decode(seq[plen:].tolist(), skip_special_tokens=True)
    else:
        text = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
    return text, f"nfe={nfe}"


def run_path_on_shard(
    path_name: str,
    shard_rows: List[tuple[int, dict]],
    args_dict: dict,
    rank: int,
) -> Dict[str, Any]:
    model_path = Path(args_dict["model_path"])
    max_new_tokens = args_dict["max_new_tokens"]
    train_temperature = args_dict["train_temperature"]
    train_seed = args_dict["train_seed"]
    max_prompt_length = args_dict["max_prompt_length"]
    mem_fraction = args_dict["mem_fraction_static"]

    from transformers import AutoModel, AutoTokenizer

    from training_aligned_sglang import DreamSGLangEngineConfig, build_dream_sglang_engine

    passed = 0
    multi_round = 0
    details: List[dict] = []
    t0 = time.time()

    engine = None
    hf_model = None
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)

    try:
        if path_name == "hf_multiblock":
            hf_model = AutoModel.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
            ).cuda().eval()
        elif path_name.startswith("sglang"):
            sgl_cfg = DreamSGLangEngineConfig(
                model_path=model_path,
                rank=rank,
                mem_fraction_static=mem_fraction,
                train_temperature=train_temperature,
            )
            engine = build_dream_sglang_engine(sgl_cfg)
            # Training loads real weights via FSDP sync; benchmark uses load_format=auto (no release).
        else:
            raise ValueError(path_name)

        for local_i, (global_idx, row) in enumerate(shard_rows):
            prompt_ids = prompt_to_ids(tokenizer, row["prompt"], max_prompt_length)

            if path_name == "hf_multiblock":
                text, nfe_log = hf_multiblock_generate(
                    hf_model,
                    tokenizer,
                    prompt_ids,
                    max_new_tokens,
                    temperature=0.0,
                    seed=None,
                )
            elif path_name == "sglang_train":
                sampling = {
                    "n": 1,
                    "top_p": 0.95,
                    "temperature": train_temperature,
                    "max_new_tokens": max_new_tokens,
                    "sampling_seed": train_seed + global_idx,
                }
                text, nfe_log = sglang_generate_text(engine, tokenizer, prompt_ids, sampling, max_new_tokens)
            elif path_name == "sglang_val":
                sampling = {
                    "top_p": 0.95,
                    "temperature": 0.0,
                    "n": 1,
                    "max_new_tokens": max_new_tokens,
                }
                text, nfe_log = sglang_generate_text(engine, tokenizer, prompt_ids, sampling, max_new_tokens)
            else:
                raise ValueError(path_name)

            if "nfe_rounds=" in nfe_log:
                multi_round += 1
            ok = score_row(row, text)
            passed += int(ok)
            details.append(
                {
                    "global_idx": global_idx,
                    "pass": ok,
                    "nfe": nfe_log,
                    "text_head": text[:200],
                }
            )
            if (local_i + 1) % 5 == 0 or local_i + 1 == len(shard_rows):
                print(
                    f"[rank{rank}][{path_name}] {local_i + 1}/{len(shard_rows)} "
                    f"pass={passed} ({100 * passed / (local_i + 1):.1f}%) "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    finally:
        if engine is not None:
            del engine
        if hf_model is not None:
            del hf_model
        gc.collect()
        torch.cuda.empty_cache()

    n = len(shard_rows)
    return {
        "path": path_name,
        "rank": rank,
        "n": n,
        "pass": passed,
        "pass_at_1": passed / n if n else 0.0,
        "multi_round_nfe_count": multi_round,
        "elapsed_s": round(time.time() - t0, 2),
        "details": details,
    }


def _apply_worker_env() -> None:
    os.environ.update(_fix_worker_env(dict(os.environ)))


def _worker(rank: int, world_size: int, args_dict: dict, paths_to_run: List[str], partial_dir: Path):
    _apply_worker_env()

    df = pd.read_parquet(args_dict["parquet"])
    offset = args_dict["offset"]
    limit = args_dict["limit"]
    all_indices = list(range(offset, min(offset + limit, len(df))))
    my_indices = all_indices[rank::world_size]

    shard_rows = [(i, df.iloc[i].to_dict()) for i in my_indices]
    print(f"[rank{rank}] shard {len(shard_rows)}/{len(all_indices)} problems paths={paths_to_run}", flush=True)

    partial: Dict[str, Any] = {"rank": rank, "world_size": world_size, "paths": {}}
    for path_name in paths_to_run:
        partial["paths"][path_name] = run_path_on_shard(path_name, shard_rows, args_dict, rank)
        partial_path = partial_dir / f"rank{rank}.{path_name}.json"
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(partial["paths"][path_name], f, ensure_ascii=False, indent=2)
        print(f"[rank{rank}] wrote {partial_path}", flush=True)


def _merge_partials(partial_dir: Path, paths: List[str], world_size: int) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for path_name in paths:
        all_details: List[dict] = []
        total_pass = 0
        total_n = 0
        multi_round = 0
        elapsed = 0.0
        for rank in range(world_size):
            p = partial_dir / f"rank{rank}.{path_name}.json"
            if not p.is_file():
                raise FileNotFoundError(f"Missing partial result: {p}")
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            total_pass += int(r["pass"])
            total_n += int(r["n"])
            multi_round += int(r.get("multi_round_nfe_count", 0))
            elapsed = max(elapsed, float(r.get("elapsed_s", 0)))
            all_details.extend(r.get("details", []))
        all_details.sort(key=lambda x: x.get("global_idx", 0))
        merged[path_name] = {
            "path": path_name,
            "n": total_n,
            "pass": total_pass,
            "pass_at_1": total_pass / total_n if total_n else 0.0,
            "multi_round_nfe_count": multi_round,
            "elapsed_s": round(elapsed, 2),
            "details": all_details,
        }
    return merged


def main():
    args = parse_args()
    if args.world_size < 1:
        raise SystemExit("--world-size must be >= 1")

    if args.worker_rank is not None:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        paths = [p.strip() for p in args.paths.split(",") if p.strip()]
        partial_dir = args.partial_dir or (DARE_ROOT / "logs/benchmarks/humaneval_partials")
        partial_dir.mkdir(parents=True, exist_ok=True)
        args_dict = {
            "model_path": str(args.model_path.resolve()),
            "parquet": str(args.parquet.resolve()),
            "offset": args.offset,
            "limit": args.limit,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "train_temperature": args.train_temperature,
            "train_seed": args.train_seed,
            "mem_fraction_static": _mem_fraction(args),
        }
        _worker(args.worker_rank, args.world_size, args_dict, paths, partial_dir)
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    n_visible = len([x for x in visible.split(",") if x.strip() != ""])
    if args.world_size > n_visible:
        print(
            f"[WARN] world_size={args.world_size} > len(CUDA_VISIBLE_DEVICES)={n_visible}; "
            "set CUDA_VISIBLE_DEVICES=0,1,2,3",
            flush=True,
        )

    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    mem_fraction = _mem_fraction(args)

    args_dict = {
        "model_path": str(args.model_path.resolve()),
        "parquet": str(args.parquet.resolve()),
        "offset": args.offset,
        "limit": args.limit,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "train_temperature": args.train_temperature,
        "train_seed": args.train_seed,
        "mem_fraction_static": mem_fraction,
    }

    partial_dir = args.partial_dir or (
        (args.output_json.parent if args.output_json else DARE_ROOT / "logs/benchmarks")
        / "humaneval_partials"
    )
    partial_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "config": {**args_dict, "paths": paths, "world_size": args.world_size, "mem_fraction_static": mem_fraction},
        "paths": {},
    }
    if args.merge_json and args.merge_json.is_file():
        with open(args.merge_json, encoding="utf-8") as f:
            prev = json.load(f)
        results["paths"] = dict(prev.get("paths", {}))

    paths_to_run = [p for p in paths if p not in results["paths"]]
    if not paths_to_run:
        print("All paths already in output; nothing to run.", flush=True)
    else:
        print(
            f"Running paths={paths_to_run} world_size={args.world_size} "
            f"mem_fraction_static={mem_fraction} partial_dir={partial_dir}",
            flush=True,
        )
        _launch_workers(args, args_dict, paths_to_run, partial_dir)
        merged = _merge_partials(partial_dir, paths_to_run, args.world_size)
        results["paths"].update(merged)

    print("\n--- summary ---", flush=True)
    for name, r in results["paths"].items():
        print(f"  {name}: pass@1={r['pass_at_1']:.3f} ({r['pass']}/{r['n']})", flush=True)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        out = dict(results)
        for name in out["paths"]:
            if "details" in out["paths"][name]:
                out["paths"][name] = {k: v for k, v in out["paths"][name].items() if k != "details"}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
