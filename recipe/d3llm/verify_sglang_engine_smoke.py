#!/usr/bin/env python3
"""Phase 3A: offline SGLang Engine smoke for d3LLM Dream (FullAttnMultiBlock).

Usage:
  unset PYTORCH_CUDA_ALLOC_CONF
  python recipe/d3llm/verify_sglang_engine_smoke.py --smoke

Requires third_party/sglang on PR #20615 branch (Dream + FullAttnMultiBlock).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

DARE_ROOT = Path(__file__).resolve().parents[2]
SGLANG_PYTHON = DARE_ROOT / "third_party" / "sglang" / "python"
if str(SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_PYTHON))

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("SGLANG_BLOCK_NONZERO_RANK_CHILDREN", "0")

DEFAULT_MODEL = DARE_ROOT / "models" / "finetune_d3LLM"
MASK_TOKEN_ID = 151666
PAD_TOKEN_ID = 151643

SMOKE_PROMPT = """<|im_start|>user
Write a Python function `add(a, b)` that returns the sum of two integers.
<|im_start|>assistant
"""


def parse_args():
    p = argparse.ArgumentParser(description="SGLang Dream Engine offline smoke (phase 3A)")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--smoke", action="store_true", help="Short generation (64 tokens)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--cache-delay-iter", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--mem-fraction-static", type=float, default=0.55)
    return p.parse_args()


def _require_pr20615():
    algo = SGLANG_PYTHON / "sglang" / "srt" / "dllm" / "algorithm" / "full_attn_multi_block.py"
    model = SGLANG_PYTHON / "sglang" / "srt" / "models" / "dream.py"
    if not algo.is_file() or not model.is_file():
        raise RuntimeError(
            "SGLang missing Dream/FullAttnMultiBlock. Checkout PR #20615:\n"
            "  cd third_party/sglang && git fetch origin pull/20615/head:pr-20615-d3llm-dream\n"
            "  git checkout pr-20615-d3llm-dream"
        )


def build_engine(model_path: Path, args):
    import yaml
    from sglang.srt.entrypoints.engine import Engine

    algo_cfg = {
        "threshold": args.threshold,
        "block_add_threshold": 0.1,
        "decoded_token_threshold": 0.95,
        "block_size": args.block_length,
        "temperature": args.temperature,
        "top_p": 0.95 if args.temperature > 0 else 1.0,
        "cache_delay_iter": args.cache_delay_iter,
        "refresh_interval": 10000,
        "early_stop": True,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(algo_cfg, f)
        algo_path = f.name

    print(f"[sglang] model={model_path}")
    print(f"[sglang] algo_cfg={algo_cfg}")
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


def main():
    args = parse_args()
    if args.smoke:
        args.max_new_tokens = 64
        args.mem_fraction_static = min(args.mem_fraction_static, 0.45)

    _require_pr20615()

    if not args.model_path.is_dir():
        raise FileNotFoundError(
            f"Model not found: {args.model_path}. Run setup_finetune_d3llm_model_code.sh"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    prompt_ids = tokenizer.encode(SMOKE_PROMPT, add_special_tokens=False)
    print(f"[info] prompt_tokens={len(prompt_ids)}")

    t0 = time.time()
    engine = build_engine(args.model_path, args)
    print(f"[info] Engine ready in {time.time() - t0:.1f}s")

    t1 = time.time()
    out = engine.generate(
        prompt=None,
        input_ids=[prompt_ids],
        sampling_params={
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": 0.95 if args.temperature > 0 else 1.0,
            "sampling_seed": 42,
        },
        return_logprob=False,
    )
    elapsed = time.time() - t1
    if isinstance(out, list):
        out = out[0] if out else {}

    output_ids = out.get("output_ids") or []
    meta = out.get("meta_info") or {}
    text = out.get("text") or tokenizer.decode(output_ids, skip_special_tokens=True)
    nfe = meta.get("nfe")
    mask_left = sum(1 for t in output_ids if t == MASK_TOKEN_ID)
    pad_left = sum(1 for t in output_ids if t == PAD_TOKEN_ID)

    print(f"[PASS] generate elapsed={elapsed:.2f}s nfe={nfe} out_tokens={len(output_ids)} "
          f"mask_in_output={mask_left} pad_in_output={pad_left}")
    print(f"[preview] {text[:400]}")

    if mask_left > 0:
        raise SystemExit("[FAIL] mask tokens remain in output")
    if len(output_ids) < 8:
        raise SystemExit("[FAIL] output too short")
    if "def add" not in text and "def " not in text:
        print("[WARN] no obvious code block in preview (may still be OK for smoke)")

    print("[PASS] phase 3A SGLang Engine smoke complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
