"""Per-rank rollout debug logs: response, extracted code, reward/acc."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import torch


def rollout_verbose_enabled(gen_kwargs: Optional[Dict[str, Any]] = None) -> bool:
    if gen_kwargs and gen_kwargs.get("rollout_verbose"):
        return True
    for key in ("DREAM_ROLLOUT_VERBOSE", "D3LLM_ROLLOUT_VERBOSE"):
        if os.environ.get(key, "0") == "1":
            return True
    return False


def format_nfe_for_log(nfe: Any) -> str:
    """Format SGLang meta_info['nfe'] for rollout logs.

    Dream + needs_full_prefill may run multiple DLLM scheduler rounds; each round
    appends one element. len(nfe)>1 usually means staging was truncated (KV/token
    budget) and rollout time scales roughly with the number of rounds.
    """
    if nfe is None:
        return "nfe=n/a"
    if isinstance(nfe, (list, tuple)):
        if len(nfe) == 0:
            return "nfe=[]"
        if len(nfe) == 1:
            return f"nfe={nfe[0]}"
        total = sum(int(x) for x in nfe)
        return (
            f"nfe_rounds={len(nfe)} nfe={list(nfe)} nfe_total={total} "
            f"(WARN: multi-round DLLM; expect ~{len(nfe)}x forward cost)"
        )
    return f"nfe={nfe}"


def rollout_log_dir() -> Optional[str]:
    for key in ("DREAM_ROLLOUT_LOG_DIR", "D3LLM_ROLLOUT_LOG_DIR"):
        path = os.environ.get(key, "").strip()
        if path:
            return path
    return None


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def rollout_detail_log_enabled() -> bool:
    """When DREAM_ROLLOUT_LOG_RANK is set, only that rank writes detailed rollout logs."""
    spec = os.environ.get("DREAM_ROLLOUT_LOG_RANK", "").strip()
    if spec == "":
        return True
    try:
        return _rank() == int(spec)
    except ValueError:
        return True


def _append_file(text: str) -> None:
    log_dir = rollout_log_dir()
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"rank{_rank()}.rollout.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    _append_file(line)


def truncate_text(text: str, max_chars: int = 1200) -> str:
    text = text.replace("\r\n", "\n")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n... [truncated] ...\n" + text[-half:]


def extract_code_preview(text: str, max_chars: int = 800) -> str:
    from verl.utils.reward_score.code_reward import extract_code_from_model

    code = extract_code_from_model(text)
    if code:
        return truncate_text(code, max_chars)
    return "(no extractable code)"


def _unwrap_non_tensor_obj(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (dict, list, str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _as_mapping(value: Any) -> Optional[Dict[str, Any]]:
    value = _unwrap_non_tensor_obj(value)
    return value if isinstance(value, dict) else None


def evaluate_code_reward(
    response_text: str,
    data_source: str,
    reward_model: Optional[Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
    *,
    rollout_nfe: Any = None,
    rollout_gen_tokens: Optional[int] = None,
    valid_response_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the same code reward path as training (best-effort, never raises)."""
    reward_model = _as_mapping(reward_model)
    if not reward_model:
        return {"skipped": True, "reason": "no reward_model"}
    ground_truth = reward_model.get("ground_truth")
    if ground_truth is None:
        return {"skipped": True, "reason": "no ground_truth"}

    extra = dict(_as_mapping(extra_info) or {})
    extra.setdefault("task", "code")

    rollout_stats: Dict[str, Any] = {}
    if rollout_nfe is not None or rollout_gen_tokens is not None or valid_response_length is not None:
        from verl.utils.reward_score.code_efficiency import compute_tpf, normalize_rollout_nfe

        nfe = normalize_rollout_nfe(rollout_nfe if rollout_nfe is not None else extra.get("rollout_nfe", 0))
        gen_tokens = rollout_gen_tokens if rollout_gen_tokens is not None else extra.get("rollout_gen_tokens")
        if gen_tokens is None or int(gen_tokens) <= 0:
            gen_tokens = int(valid_response_length or 0)
        else:
            gen_tokens = int(gen_tokens)
        tpf = compute_tpf(gen_tokens, nfe)
        rollout_stats = {
            "rollout_nfe": nfe,
            "rollout_gen_tokens": gen_tokens,
            "tpf": tpf,
        }
        extra.update(rollout_stats)

    try:
        from verl.utils.reward_score import dllm_rm

        result = dllm_rm(
            data_source=data_source,
            solution_str=response_text,
            ground_truth=ground_truth,
            extra_info=extra,
        )
        if isinstance(result, dict):
            metadata = {k: v for k, v in result.items() if k not in ("score", "acc", "pred")}
            for key, value in rollout_stats.items():
                metadata[key] = value
            return {
                "reward": float(result.get("score", result.get("reward", 0.0))),
                "is_correct": bool(result.get("acc", result.get("is_correct", False))),
                "pred_preview": truncate_text(str(result.get("pred", "")), 600),
                "metadata": metadata,
            }
        return {"reward": float(result), "is_correct": bool(result > 0)}
    except Exception as exc:
        return {"error": str(exc)}


def build_sample_meta(
    prompts,
    batch_size: int,
    n_rollout: int,
    tokenizer,
    non_tensor_batch: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    meta_list: List[Dict[str, Any]] = []
    nt = non_tensor_batch if non_tensor_batch is not None else (getattr(prompts, "non_tensor_batch", None) or {})

    def _repeat_field(key: str, default: Any = None):
        if key not in nt:
            return [default] * batch_size
        arr = nt[key]
        base = batch_size // n_rollout if n_rollout else batch_size
        if hasattr(arr, "__len__") and len(arr) == base:
            out = []
            for v in arr:
                out.extend([v] * n_rollout)
            return out
        if hasattr(arr, "__len__") and len(arr) == batch_size:
            return list(arr)
        return [default] * batch_size

    uids = _repeat_field("uid", None)
    sources = _repeat_field("data_source", "?")
    extras = _repeat_field("extra_info", {})
    indices = _repeat_field("index", None)
    raw_prompts = _repeat_field("raw_prompt", None)
    reward_models = _repeat_field("reward_model", None)
    rollout_nfes = _repeat_field("rollout_nfe", 0)
    rollout_gen_tokens = _repeat_field("rollout_gen_tokens", 0)

    for i in range(batch_size):
        extra = _as_mapping(extras[i]) or {}
        task = extra.get("task", "?") if extra else "?"
        prompt_preview = ""
        rp = raw_prompts[i]
        if rp is not None:
            try:
                if isinstance(rp, list):
                    prompt_preview = tokenizer.apply_chat_template(
                        rp, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt_preview = str(rp)
            except Exception:
                prompt_preview = str(rp)[:400]

        meta_list.append(
            {
                "uid": uids[i] if uids[i] is not None else indices[i],
                "index": indices[i],
                "data_source": sources[i],
                "task": task,
                "prompt_preview": prompt_preview,
                "extra_info": extra,
                "reward_model": _as_mapping(reward_models[i]),
                "rollout_nfe": rollout_nfes[i],
                "rollout_gen_tokens": rollout_gen_tokens[i],
            }
        )
    return meta_list


def _format_reward_log_line(reward_info: Dict[str, Any]) -> str:
    if reward_info.get("skipped"):
        return f"n/a (skipped: {reward_info.get('reason', '?')})"
    if "error" in reward_info:
        err = str(reward_info["error"])
        if len(err) > 120:
            err = err[:120] + "..."
        return f"n/a (error: {err})"
    acc = reward_info.get("is_correct", "n/a")
    return f"reward={reward_info.get('reward', 'n/a')} acc={acc}"


def log_rollout_batch(
    *,
    prompts,
    responses: torch.Tensor,
    idx_repeat: torch.Tensor,
    gen_kwargs: Dict[str, Any],
    tokenizer,
    elapsed_s: float,
    is_validate: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
    non_tensor_batch: Optional[Dict[str, Any]] = None,
) -> None:
    if not rollout_verbose_enabled(gen_kwargs):
        return
    if not rollout_detail_log_enabled():
        return

    batch_size = responses.size(0)
    prompt_length = idx_repeat.size(1)
    prompt_bs = prompts.batch["input_ids"].size(0)
    n_rollout = max(1, batch_size // prompt_bs)
    step = prompts.meta_info.get("global_step", gen_kwargs.get("global_step", "?"))
    mode = "val" if is_validate else "train"
    rank = _rank()

    sample_meta = build_sample_meta(
        prompts,
        batch_size,
        n_rollout,
        tokenizer,
        non_tensor_batch=non_tensor_batch,
    )
    n_correct = 0
    n_scored = 0
    n_errors = 0
    rewards: List[float] = []
    prompt_length = idx_repeat.size(1)

    log(
        f"[dream][RANK{rank}] batch_start step={step} mode={mode} "
        f"n_prompts={prompt_bs} n_rollout={n_rollout} n_samples={batch_size} "
        f"dllm_decode={gen_kwargs.get('dllm_decode', 'entropy')} elapsed={elapsed_s:.2f}s"
    )

    per_sample_records: List[Dict[str, Any]] = []

    for i in range(batch_size):
        meta = sample_meta[i] if i < len(sample_meta) else {}
        uid = meta.get("uid", meta.get("index", i))
        ds = meta.get("data_source", "?")
        response_ids = responses[i]
        valid_len = responses[i].numel()
        if attention_mask is not None:
            valid_len = int(attention_mask[i, prompt_length:].sum().item())
            response_ids = response_ids[:valid_len]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        reward_info = evaluate_code_reward(
            response_text=response_text,
            data_source=ds,
            reward_model=meta.get("reward_model"),
            extra_info=meta.get("extra_info"),
            rollout_nfe=meta.get("rollout_nfe"),
            rollout_gen_tokens=meta.get("rollout_gen_tokens"),
            valid_response_length=valid_len,
        )
        if not reward_info.get("skipped") and "error" not in reward_info:
            n_scored += 1
            rewards.append(float(reward_info.get("reward", 0.0)))
            if reward_info.get("is_correct"):
                n_correct += 1
        elif "error" in reward_info:
            n_errors += 1

        rollout_idx = i % n_rollout if n_rollout else 0
        log(
            f"[dream][RANK{rank}] sample[{i}] prompt_idx={i // n_rollout if n_rollout else i} "
            f"rollout_idx={rollout_idx} uid={uid} data_source={ds} "
            f"{_format_reward_log_line(reward_info)}"
        )
        if reward_info.get("pred_preview"):
            log(f"[dream][RANK{rank}] sample[{i}] extracted_code:\n{reward_info['pred_preview']}")
        else:
            log(f"[dream][RANK{rank}] sample[{i}] code_preview:\n{extract_code_preview(response_text)}")
        log(f"[dream][RANK{rank}] sample[{i}] response_text:\n{truncate_text(response_text)}")
        if reward_info.get("metadata"):
            try:
                meta_str = json.dumps(reward_info["metadata"], ensure_ascii=False, default=str)
            except Exception:
                meta_str = str(reward_info["metadata"])
            log(f"[dream][RANK{rank}] sample[{i}] test_metadata: {truncate_text(meta_str, 800)}")

        per_sample_records.append(
            {
                "i": i,
                "prompt_idx": i // n_rollout if n_rollout else i,
                "rollout_idx": rollout_idx,
                "uid": uid,
                "data_source": ds,
                "reward_info": reward_info,
            }
        )

    # Train: one line per prompt summarizing all n_rollout responses (e.g. 4/4 acc).
    if n_rollout > 1 and mode == "train" and per_sample_records:
        for pidx in range(prompt_bs):
            group = [r for r in per_sample_records if r["prompt_idx"] == pidx]
            if not group:
                continue
            g_rewards: List[float] = []
            g_correct = 0
            g_scored = 0
            for r in group:
                ri = r["reward_info"]
                if ri.get("skipped") or "error" in ri:
                    continue
                g_scored += 1
                rew = float(ri.get("reward", 0.0))
                g_rewards.append(rew)
                if ri.get("is_correct"):
                    g_correct += 1
            uid0 = group[0]["uid"]
            ds0 = group[0]["data_source"]
            if g_scored:
                avg_r = sum(g_rewards) / len(g_rewards)
                log(
                    f"[dream][RANK{rank}] prompt_group step={step} mode={mode} "
                    f"prompt_idx={pidx} uid={uid0} data_source={ds0} "
                    f"n_rollout={len(group)} rollout_acc={g_correct}/{g_scored} "
                    f"({100.0 * g_correct / g_scored:.1f}%) avg_reward={avg_r:.4f}"
                )
            else:
                log(
                    f"[dream][RANK{rank}] prompt_group step={step} mode={mode} "
                    f"prompt_idx={pidx} uid={uid0} data_source={ds0} n_rollout={len(group)} (not scored)"
                )
            for r in group:
                ri = r["reward_info"]
                log(
                    f"[dream][RANK{rank}]   rollout[{r['rollout_idx']}] sample[{r['i']}] "
                    f"{_format_reward_log_line(ri)}"
                )

    if n_scored:
        avg_reward = sum(rewards) / len(rewards)
        log(
            f"[dream][RANK{rank}] batch_done step={step} mode={mode} "
            f"n_samples={batch_size} n_scored={n_scored} "
            f"batch_acc={n_correct}/{n_scored} ({100.0 * n_correct / n_scored:.1f}%) "
            f"avg_reward={avg_reward:.4f}"
        )
    else:
        reason = "no code reward scored"
        if n_errors:
            reason += f"; {n_errors} eval error(s) (check data_source/extra_info on gen batch)"
        log(
            f"[dream][RANK{rank}] batch_done step={step} mode={mode} "
            f"n_samples={batch_size} ({reason})"
        )
