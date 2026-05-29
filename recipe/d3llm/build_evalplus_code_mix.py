#!/usr/bin/env python3
"""Build EvalPlus-aligned code RL parquets (train mix + val humaneval)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DARE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DARE_ROOT / "recipe" / "d3llm"))

from evalplus_prompt import (  # noqa: E402
    convert_row_to_evalplus,
    extract_taco_codeblock,
    format_evalplus_messages,
    _as_list,
)

RL_DIR = DARE_ROOT / "data" / "preprocessed" / "rl"
DEFAULT_TRAIN_OUT = RL_DIR / "train" / "code_evalplus_mix_1.parquet"
DEFAULT_VAL_OUT = RL_DIR / "test" / "humaneval_evalplus_1.parquet"


def _load_parquet(path: Path) -> List[Dict[str, Any]]:
    return pd.read_parquet(path).to_dict(orient="records")


def _write_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"Wrote {path} ({len(rows)} rows)")


def convert_parquet(
    input_path: Path,
    *,
    split: str,
    row_filter=None,
    max_rows: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    records = _load_parquet(input_path)
    if row_filter is not None:
        records = [r for r in records if row_filter(r)]
    if max_rows is not None and len(records) > max_rows:
        rng = random.Random(seed)
        records = rng.sample(records, max_rows)

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(records):
        converted = convert_row_to_evalplus(row, index=idx, split=split)
        if converted is not None:
            out.append(converted)
    return out


def is_taco_completion_row(row: Dict[str, Any]) -> bool:
    prompt = _as_list(row.get("prompt"))
    if not prompt:
        return False
    content = prompt[0].get("content", "")
    return extract_taco_codeblock(content) is not None


def convert_taco_completion_row(row: Dict[str, Any], *, index: int, split: str) -> Optional[Dict[str, Any]]:
    content = _as_list(row.get("prompt"))[0]["content"]
    task_body = extract_taco_codeblock(content)
    if not task_body:
        return None
    return {
        "data_source": row["data_source"],
        "prompt": format_evalplus_messages(task_body, include_assistant_prefix=True),
        "reward_model": row["reward_model"],
        "extra_info": {
            **(row.get("extra_info") or {}),
            "split": split,
            "index": index,
            "prompt_style": "evalplus",
            "taco_style": "completion",
        },
    }


def build_train_mix(
    *,
    competition_cap: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0

    # Completion-style: MBPP (train on test split; val is humaneval only)
    mbpp_rows = convert_parquet(RL_DIR / "test" / "mbpp_1.parquet", split="train")
    for r in mbpp_rows:
        r["extra_info"]["index"] = idx
        r["extra_info"]["mix_bucket"] = "mbpp_completion"
        rows.append(r)
        idx += 1
    print(f"  mbpp completion: {len(mbpp_rows)}")

    # TACO: all code-fence completion samples
    taco_all = _load_parquet(RL_DIR / "train" / "taco-K8_1.parquet")
    taco_completion = [r for r in taco_all if is_taco_completion_row(r)]
    for row in taco_completion:
        converted = convert_taco_completion_row(row, index=idx, split="train")
        if converted:
            converted["extra_info"]["mix_bucket"] = "taco_completion"
            rows.append(converted)
            idx += 1
    print(f"  taco completion: {len(taco_completion)}")

    # Competition-style: capped LCB + PrimeIntellect + remaining TACO
    rng = random.Random(seed)
    lcb_rows = convert_parquet(RL_DIR / "train" / "lcbv5-K8_1.parquet", split="train")
    for r in lcb_rows:
        r["extra_info"]["index"] = idx
        r["extra_info"]["mix_bucket"] = "lcbv5_competition"
        rows.append(r)
        idx += 1
    print(f"  lcbv5 competition: {len(lcb_rows)}")

    pi_cap = min(competition_cap, len(_load_parquet(RL_DIR / "train" / "primeintellect-K8_1.parquet")))
    pi_rows = convert_parquet(
        RL_DIR / "train" / "primeintellect-K8_1.parquet",
        split="train",
        max_rows=pi_cap,
        seed=seed,
    )
    for r in pi_rows:
        r["extra_info"]["index"] = idx
        r["extra_info"]["mix_bucket"] = "primeintellect_competition"
        rows.append(r)
        idx += 1
    print(f"  primeintellect competition (cap {pi_cap}): {len(pi_rows)}")

    taco_rest = [r for r in taco_all if not is_taco_completion_row(r)]
    taco_comp_cap = min(competition_cap, len(taco_rest))
    taco_comp = rng.sample(taco_rest, taco_comp_cap) if taco_comp_cap else []
    for row in taco_comp:
        converted = convert_row_to_evalplus(row, index=idx, split="train")
        if converted:
            converted["extra_info"]["mix_bucket"] = "taco_competition"
            rows.append(converted)
            idx += 1
    print(f"  taco competition (cap {taco_comp_cap}): {len(taco_comp)}")

    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["extra_info"]["index"] = i
    return rows


def validate_row(row: Dict[str, Any], tokenizer=None) -> List[str]:
    errors: List[str] = []
    prompt = _as_list(row.get("prompt"))
    if len(prompt) < 2:
        errors.append("prompt must be [user, assistant] for EvalPlus alignment")
        return errors
    if prompt[0].get("role") != "user" or prompt[1].get("role") != "assistant":
        errors.append("prompt roles must be user then assistant")
    user = prompt[0].get("content", "")
    if "Please provide a self-contained Python script" not in user:
        errors.append("user content missing EvalPlus instruction prefix")
    if "```\n" not in user or user.rstrip().endswith("```") is False:
        errors.append("user content missing task ``` fence")
    if row.get("data_source") == "mbpp":
        if "Your code should pass these tests:" not in user:
            errors.append("mbpp user content missing test assertions")
        if "assert " not in user:
            errors.append("mbpp user content missing assert lines")
    assistant = prompt[1].get("content", "")
    if "Below is a Python script" not in assistant or not assistant.endswith("```python\n"):
        errors.append("assistant content missing EvalPlus response prefix")
    rm = row.get("reward_model")
    if not isinstance(rm, dict) or "ground_truth" not in rm:
        errors.append("reward_model.ground_truth missing")
    if tokenizer is not None:
        try:
            text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            if "Please provide a self-contained Python script" not in text:
                errors.append("chat_template output missing instruction prefix")
        except Exception as exc:  # pragma: no cover
            errors.append(f"chat_template failed: {exc}")
    return errors


def validate_parquet(path: Path, tokenizer=None, sample: int = 5) -> None:
    df = pd.read_parquet(path)
    print(f"\nValidate {path}: {len(df)} rows")
    buckets = {}
    for _, row in df.iterrows():
        b = (row.get("extra_info") or {}).get("mix_bucket", row.get("data_source"))
        buckets[b] = buckets.get(b, 0) + 1
    print("  buckets:", buckets)
    print("  data_sources:", df["data_source"].value_counts().to_dict())

    errors_total = 0
    for i in range(min(sample, len(df))):
        row = df.iloc[i].to_dict()
        errs = validate_row(row, tokenizer=tokenizer)
        if errs:
            errors_total += len(errs)
            print(f"  row {i} errors: {errs}")
    if errors_total:
        raise SystemExit(f"Validation failed with {errors_total} error(s)")
    print("  sample validation OK")


def dump_review_samples(
    rows: List[Dict[str, Any]],
    out_path: Path,
    *,
    uids: Optional[List[int]] = None,
    per_bucket: int = 2,
) -> None:
    """Write human-readable JSONL samples for manual review."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected: List[Dict[str, Any]] = []
    if uids:
        uid_set = set(uids)
        for row in rows:
            idx = (row.get("extra_info") or {}).get("index")
            if idx in uid_set:
                selected.append(row)
    buckets: Dict[str, int] = {}
    for row in rows:
        if uids and (row.get("extra_info") or {}).get("index") in set(uids):
            continue
        bucket = (row.get("extra_info") or {}).get("mix_bucket", row.get("data_source", "?"))
        if buckets.get(bucket, 0) >= per_bucket:
            continue
        selected.append(row)
        buckets[bucket] = buckets.get(bucket, 0) + 1

    with out_path.open("w", encoding="utf-8") as f:
        for row in selected:
            prompt = _as_list(row.get("prompt"))
            user = prompt[0]["content"] if prompt else ""
            assistant = prompt[1]["content"] if len(prompt) > 1 else ""
            gt = (row.get("reward_model") or {}).get("ground_truth")
            f.write(
                json.dumps(
                    {
                        "index": (row.get("extra_info") or {}).get("index"),
                        "data_source": row.get("data_source"),
                        "mix_bucket": (row.get("extra_info") or {}).get("mix_bucket"),
                        "entry_point": (row.get("extra_info") or {}).get("entry_point"),
                        "user_prompt": user,
                        "assistant_prefix": assistant,
                        "ground_truth_tests": gt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote review samples: {out_path} ({len(selected)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EvalPlus-aligned code RL parquets")
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--val-out", type=Path, default=DEFAULT_VAL_OUT)
    parser.add_argument("--competition-cap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="models/finetune_d3LLM")
    parser.add_argument("--samples-out", type=Path, default=RL_DIR / "train" / "code_evalplus_mix_samples.jsonl")
    parser.add_argument("--sample-uids", type=str, default="208,669,150,356,414")
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    args = parser.parse_args()

    print("Building EvalPlus val (humaneval)...")
    val_rows = convert_parquet(RL_DIR / "test" / "humaneval_1.parquet", split="test")
    for i, r in enumerate(val_rows):
        r["extra_info"]["index"] = i
    _write_parquet(val_rows, args.val_out)

    print("Building EvalPlus train mix...")
    train_rows = build_train_mix(competition_cap=args.competition_cap, seed=args.seed)
    _write_parquet(train_rows, args.train_out)

    tokenizer = None
    if not args.skip_tokenizer_check:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:
            print(f"[WARN] tokenizer load failed ({exc}); skipping chat_template check")

    validate_parquet(args.val_out, tokenizer=tokenizer)
    validate_parquet(args.train_out, tokenizer=tokenizer)

    sample_uids = [int(x.strip()) for x in args.sample_uids.split(",") if x.strip()]
    dump_review_samples(train_rows, args.samples_out, uids=sample_uids, per_bucket=2)

    summary = {
        "train_out": str(args.train_out),
        "val_out": str(args.val_out),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
