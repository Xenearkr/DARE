#!/usr/bin/env python3
"""
Phase-0 check: load models/finetune_d3LLM offline and run Dream / d3LLM multiblock generation.

Does not import or modify verl training code.

Usage:
  export D3LLM_ROOT=/path/to/d3LLM
  export HF_HUB_OFFLINE=1
  python recipe/d3llm/verify_finetune_d3llm.py
  python recipe/d3llm/verify_finetune_d3llm.py --mode multiblock --max-new-tokens 128
  python recipe/d3llm/verify_finetune_d3llm.py --lora-path path/to/adapter  # optional LoRA smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# recipe/d3llm -> DARE root
DARE_ROOT = Path(__file__).resolve().parents[2]
RECIPE_D3LLM = Path(__file__).resolve().parent
DEFAULT_MODEL = DARE_ROOT / "models" / "finetune_d3LLM"

if str(RECIPE_D3LLM) not in sys.path:
    sys.path.insert(0, str(RECIPE_D3LLM))

from d3llm_multiblock import D3LLMMultiBlockConfig, bind_multiblock, ensure_d3llm_on_path


CODE_PROMPT = """<|im_start|>user
Write a Python function `add(a, b)` that returns the sum of two integers.
<|im_start|>assistant
"""


def _first_sequence(out) -> torch.Tensor:
    # diffusion_generate returns (DreamModelOutput, nfe) for Dream / d3LLM wrappers
    if isinstance(out, (tuple, list)) and len(out) > 0:
        out = out[0]
    if hasattr(out, "sequences"):
        seq = out.sequences
    else:
        seq = out
    if not isinstance(seq, torch.Tensor):
        raise TypeError(f"Expected tensor sequences, got {type(seq)}")
    if seq.dim() == 2:
        seq = seq[0]
    return seq


def _decode_output(tokenizer, out) -> str:
    ids = _first_sequence(out)
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


def parse_args():
    p = argparse.ArgumentParser(description="Verify finetune_d3LLM loads and generates (phase 0).")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--d3llm-root", type=Path, default=None)
    p.add_argument(
        "--mode",
        choices=("load", "vanilla", "multiblock", "all"),
        default="all",
        help="load: weights only; vanilla: Dream diffusion_generate; multiblock: d3LLM bind",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lora-path", type=Path, default=None, help="Optional PEFT adapter dir for LoRA smoke test")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    return p.parse_args()


def load_model(model_path: Path, dtype: str, lora_path: Path | None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch_dtype = getattr(torch, dtype)
    print(f"[1/4] Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)

    print(f"[2/4] Loading weights (trust_remote_code, local_files_only) ...")
    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=True,
    )

    if lora_path is not None:
        from peft import PeftModel

        print(f"[2b] Loading LoRA adapter from {lora_path} ...")
        model = PeftModel.from_pretrained(model, str(lora_path), is_trainable=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Loaded OK. parameters={n_params:,} dtype={dtype}")
    return tokenizer, model


def run_vanilla(model, tokenizer, device, max_new_tokens: int):
    print("[3/4] Vanilla Dream diffusion_generate (entropy) ...")
    model = model.to(device).eval()
    inputs = tokenizer(CODE_PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            steps=max_new_tokens,
            temperature=0.0,
            alg="entropy",
            return_dict_in_generate=True,
        )
    text = _decode_output(tokenizer, out)
    print("      --- vanilla output (tail) ---")
    print(text[-800:])
    return text


def run_multiblock(model, tokenizer, device, max_new_tokens: int, d3llm_root: Path | None):
    print("[4/4] d3LLM multi-block (entropy_threshold) ...")
    ensure_d3llm_on_path(str(d3llm_root) if d3llm_root else None)
    cfg = D3LLMMultiBlockConfig(max_new_tokens=max_new_tokens)
    bind_multiblock(model, cfg=cfg, d3llm_root=str(d3llm_root) if d3llm_root else None)
    model = model.to(device).eval()

    inputs = tokenizer(CODE_PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out, nfe = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            threshold=cfg.threshold,
            block_length=cfg.block_length,
        )
    text = _decode_output(tokenizer, out)
    print(f"      NFE (forward passes): {nfe}")
    print("      --- multiblock output (tail) ---")
    print(text[-800:])
    return text


def main():
    args = parse_args()
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        sys.exit(f"Model directory not found: {model_path}")

    for name in ("configuration_dream.py", "modeling_dream.py", "generation_utils.py"):
        if not (model_path / name).exists():
            sys.exit(
                f"Missing {name} under {model_path}. "
                f"Run: bash recipe/d3llm/setup_finetune_d3llm_model_code.sh"
            )

    tokenizer, model = load_model(model_path, args.dtype, args.lora_path)

    if args.mode in ("load", "all"):
        print("[OK] load-only check passed.")

    if args.mode in ("vanilla", "all"):
        run_vanilla(model, tokenizer, args.device, args.max_new_tokens)

    if args.mode in ("multiblock", "all"):
        # Reload fresh model for multiblock to avoid method shadowing from vanilla hooks
        if args.mode == "all":
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tokenizer, model = load_model(model_path, args.dtype, args.lora_path)
        run_multiblock(model, tokenizer, args.device, args.max_new_tokens, args.d3llm_root)

    print("\n[PASS] Phase-0 verification completed.")


if __name__ == "__main__":
    main()
