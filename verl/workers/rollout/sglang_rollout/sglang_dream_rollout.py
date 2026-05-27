# Copyright 2025 Shanghai AI Lab Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGLang rollout for Dream / d3LLM Dream-Coder (FullAttnMultiBlock)."""

import asyncio
import logging
import os
import random
import time
from typing import TYPE_CHECKING, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.dream_rollout_debug import (
    build_sample_meta,
    format_nfe_for_log,
    log,
    log_rollout_batch,
    rollout_verbose_enabled,
)
from verl.workers.rollout.sglang_rollout.sglang_rollout import (
    SGLangRollout,
    _post_process_outputs,
    _pre_process_inputs,
)
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _find_subseq_positions(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle or len(haystack) < len(needle):
        return []
    hits: List[int] = []
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if list(haystack[i : i + n]) == list(needle):
            hits.append(i)
    return hits


def _dream_stop_token_ids(
    tokenizer: "PreTrainedTokenizer",
    eos_token_id: Union[int, List[int], None],
    pad_token_id: int,
    mask_token_id: int,
) -> Set[int]:
    """Match SGLang stop handling used on the val path (pad/eos/im_end/mask)."""
    stops = {int(pad_token_id), int(mask_token_id)}
    if eos_token_id is not None:
        if isinstance(eos_token_id, (list, tuple)):
            stops.update(int(x) for x in eos_token_id)
        else:
            stops.add(int(eos_token_id))
    for token in ("<|endoftext|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            stops.add(int(tid))
    return stops


def _truncate_ids_like_sglang_stop(
    token_ids: List[int],
    stop_ids: Set[int],
    finish_reason: Optional[dict],
    opening_prefix_ids: Optional[List[int]],
) -> List[int]:
    """Mirror ``output_ids_through_stop``: cut at first stop token; trim tail re-openings."""
    cut = len(token_ids)
    fr_type = (finish_reason or {}).get("type")
    if fr_type == "stop":
        # Engine already applied output_ids_through_stop.
        return token_ids

    for j, tok in enumerate(token_ids):
        if int(tok) in stop_ids:
            cut = j + 1
            break

    # Train often ends with finish_reason=length (full max_new_tokens). Val ends earlier on pad.
    if opening_prefix_ids and len(opening_prefix_ids) >= 4:
        hits = _find_subseq_positions(token_ids[:cut], opening_prefix_ids)
        if len(hits) >= 2 and hits[1] > 0:
            cut = min(cut, hits[1])

    return token_ids[:cut]


def _finalize_dream_response_tensor(
    response: torch.Tensor,
    rollout_log_probs: Optional[torch.Tensor],
    stop_ids: Set[int],
    pad_token_id: int,
    response_length: int,
    finish_reasons: Optional[List[Optional[dict]]] = None,
    opening_prefix_ids: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Val path: no mask-strip; truncate at stop tokens then right-pad (same as parent SGLangRollout)."""
    from torch.nn.utils.rnn import pad_sequence

    rows: List[torch.Tensor] = []
    lp_rows: List[torch.Tensor] = []
    for i in range(response.size(0)):
        fr = finish_reasons[i] if finish_reasons and i < len(finish_reasons) else None
        ids = _truncate_ids_like_sglang_stop(
            response[i].tolist(),
            stop_ids,
            fr,
            opening_prefix_ids,
        )
        rows.append(
            torch.tensor(ids, dtype=response.dtype, device=response.device)
            if ids
            else torch.zeros(0, dtype=response.dtype, device=response.device)
        )
        if rollout_log_probs is not None:
            lp_rows.append(rollout_log_probs[i, : len(ids)])

    padded = pad_sequence(rows, batch_first=True, padding_value=pad_token_id)
    if padded.size(1) < response_length:
        padded = pad_sequence_to_length(padded, response_length, pad_token_id)
    elif padded.size(1) > response_length:
        padded = padded[:, :response_length]

    padded_lp = None
    if rollout_log_probs is not None:
        padded_lp = pad_sequence(lp_rows, batch_first=True, padding_value=0.0)
        if padded_lp.size(1) < response_length:
            padded_lp = pad_sequence_to_length(padded_lp, response_length, 0.0)
        elif padded_lp.size(1) > response_length:
            padded_lp = padded_lp[:, :response_length]

    return padded, padded_lp


class SGLangDreamRollout(SGLangRollout):
    """SGLang rollout for ``model.name=dream`` with ``dllm_decode=multiblock``."""

    def __init__(
        self,
        actor_module: str,
        config: DictConfig,
        tokenizer: "PreTrainedTokenizer",
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        self._mask_token_id = config.get("mask_token_id", 151666)
        self._block_length = config.get("block_length", 32)
        self._dllm_algorithm = config.get("dllm_algorithm", "FullAttnMultiBlock")
        self._threshold = config.get("d3llm_threshold", config.get("dllm_confidence_threshold", 0.5))
        self._block_add_threshold = config.get("d3llm_block_add_threshold", 0.1)
        self._decoded_token_threshold = config.get("d3llm_decoded_token_threshold", 0.95)
        self._per_sample_seed = bool(config.get("per_sample_seed", True))
        self._base_seed = int(config.get("base_seed", 42))

        logger.info(
            "SGLangDreamRollout: algorithm=%s mask=%s block_length=%s threshold=%s",
            self._dllm_algorithm,
            self._mask_token_id,
            self._block_length,
            self._threshold,
        )

        super().__init__(
            actor_module=actor_module,
            config=config,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            port=port,
            trust_remote_code=trust_remote_code,
            device_mesh=device_mesh,
            **kwargs,
        )

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        import tempfile

        import yaml
        from sglang.srt.entrypoints.engine import Engine

        from .utils import broadcast_pyobj, get_ip, get_open_port

        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group(self._tp_mesh_name),
                src=self._device_mesh_cpu[self._tp_mesh_name].mesh[0].item(),
                force_cpu_device=False,
            )
            from verl.workers.rollout.sglang_rollout.utils import is_ipv6

            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            rank = self._rank
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

            max_prompt = int(self.config.get("prompt_length", 1024))
            max_response = int(self.config.get("response_length", 512))
            max_seq_tokens = max_prompt + max_response + 64
            max_prefill_tokens = int(
                self.config.get("max_num_batched_tokens", max_seq_tokens)
            )
            max_prefill_tokens = max(max_prefill_tokens, max_seq_tokens)

            engine_kwargs = dict(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.get("mem_fraction_static", 0.6),
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                port=30000 + rank,
                max_running_requests=self.config.get("max_running_requests", 1),
                max_prefill_tokens=max_prefill_tokens,
                # Avoid chunked prefill splitting Dream full-length DLLM extends.
                chunked_prefill_size=-1,
            )

            engine_kwargs["dllm_algorithm"] = self._dllm_algorithm
            algo_cfg = {
                "threshold": float(self._threshold),
                "block_add_threshold": float(self._block_add_threshold),
                "decoded_token_threshold": float(self._decoded_token_threshold),
                "block_size": int(self._block_length),
                "temperature": float(self.config.get("temperature", 1.0)),
                "top_p": float(self.config.get("top_p", 1.0)),
                "cache_delay_iter": int(self.config.get("d3llm_cache_delay_iter", 32)),
                "refresh_interval": int(self.config.get("d3llm_refresh_interval", 10000)),
                "early_stop": bool(self.config.get("d3llm_early_stop", True)),
            }
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(algo_cfg, f)
                engine_kwargs["dllm_algorithm_config"] = f.name

            if self.config.get("disable_cuda_graph") is not None:
                engine_kwargs["disable_cuda_graph"] = self.config.get("disable_cuda_graph")
            if self.config.get("attention_backend"):
                engine_kwargs["attention_backend"] = self.config.get("attention_backend")

            logger.info("SGLang Engine for Dream: dllm_algorithm=%s algo_cfg=%s", self._dllm_algorithm, algo_cfg)
            self._engine = Engine(**engine_kwargs)
        else:
            self._engine = None

        self.sharding_manager = None
        if self._tp_rank == 0:
            self._engine.release_memory_occupation()
        self.is_sleep = True

    def _opening_prefix_ids(self) -> List[int]:
        return self.tokenizer.encode("To solve this problem", add_special_tokens=False)

    def _dream_sampling_extras(self, stop_ids: Set[int]) -> dict:
        return {
            "stop_token_ids": sorted(stop_ids),
            "skip_special_tokens": self.sampling_params.get("skip_special_tokens", True),
            "spaces_between_special_tokens": self.sampling_params.get(
                "spaces_between_special_tokens", True
            ),
        }

    def _apply_dream_response_finalize(
        self,
        data: DataProto,
        eos_token_id: Union[int, List[int], None],
        finish_reasons: Optional[List[Optional[dict]]] = None,
    ) -> DataProto:
        stop_ids = _dream_stop_token_ids(
            self.tokenizer, eos_token_id, self.pad_token_id, self._mask_token_id
        )
        response = data.batch["responses"]
        rollout_log_probs = data.batch.get("rollout_log_probs")
        response, rollout_log_probs = _finalize_dream_response_tensor(
            response,
            rollout_log_probs,
            stop_ids,
            self.pad_token_id,
            self.config.response_length,
            finish_reasons=finish_reasons,
            opening_prefix_ids=self._opening_prefix_ids(),
        )
        prompts = data.batch["prompts"]
        prompt_width = prompts.size(1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=data.batch["attention_mask"].dtype
        )
        attention_mask_out = torch.cat(
            (data.batch["attention_mask"][:, :prompt_width], response_attention_mask),
            dim=-1,
        )
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=data.batch["position_ids"].device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(response.size(0), -1)
        response_position_ids = data.batch["position_ids"][:, -1:] + delta_position_id
        position_ids_out = torch.cat(
            (data.batch["position_ids"][:, :prompt_width], response_position_ids),
            dim=-1,
        )
        data.batch["responses"] = response
        data.batch["attention_mask"] = attention_mask_out
        data.batch["position_ids"] = position_ids_out
        data.batch["input_ids"] = torch.cat([prompts, response], dim=-1)
        if rollout_log_probs is not None:
            data.batch["rollout_log_probs"] = rollout_log_probs
        return data

    @torch.no_grad()
    def _batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)
        eos_token_id = prompts.meta_info.get("eos_token_id", self.eos_token_id)
        if eos_token_id is None:
            eos_token_id = self.pad_token_id

        if not do_sample or is_validate:
            out = super()._batch_level_generate_sequences(prompts, **kwargs)
            return self._apply_dream_response_finalize(out, eos_token_id)

        return self._per_sample_generate_sequences(prompts, **kwargs)

    @torch.no_grad()
    def _per_sample_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """One async generate per sample (independent seeds), aligned with SDAR lmdeploy-style."""
        t_batch = time.time()
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        eos_token_id = prompts.meta_info.get("eos_token_id", self.eos_token_id)
        if eos_token_id is None:
            eos_token_id = self.pad_token_id

        batch_size = idx.size(0)
        n_rollout = self.config.n
        verbose = rollout_verbose_enabled() or bool(self.config.get("rollout_verbose", False))

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )

        raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        image_data = non_tensor_batch.pop("multi_modal_data", None) if "multi_modal_data" in non_tensor_batch else None

        idx_list = []
        image_list = []
        for i in range(batch_size):
            prompt_ids = raw_prompt_ids[i]
            if isinstance(prompt_ids, np.ndarray):
                prompt_ids = prompt_ids.tolist()
            for _ in range(n_rollout):
                idx_list.append(prompt_ids)
                if image_data is not None:
                    item = image_data[i]
                    image_list.append(item.get("image", None) if isinstance(item, dict) else None)
                else:
                    image_list.append(None)

        idx_repeat = idx.repeat_interleave(n_rollout, dim=0)
        attention_mask_repeat = attention_mask.repeat_interleave(n_rollout, dim=0)
        position_ids_repeat = position_ids.repeat_interleave(n_rollout, dim=0)
        total_batch_size = batch_size * n_rollout

        _non_tensor_batch = {}
        for key, val in non_tensor_batch.items():
            _non_tensor_batch[key] = np.repeat(val, n_rollout, axis=0)

        sample_meta = build_sample_meta(prompts, total_batch_size, n_rollout, self.tokenizer) if verbose else None

        stop_ids = _dream_stop_token_ids(
            self.tokenizer, eos_token_id, self.pad_token_id, self._mask_token_id
        )
        gen_kwargs = dict(
            n=1,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            max_new_tokens=self.config.response_length,
        )

        if self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            merged_output = []
            for j, (input_ids, image) in enumerate(zip(idx_list, image_list)):
                t0 = time.time()
                if self._per_sample_seed and self.config.temperature > 0:
                    seed = self._base_seed + j
                else:
                    seed = random.randint(0, 2**31 - 1)
                per_call_params = {
                    **self.sampling_params,
                    **gen_kwargs,
                    **self._dream_sampling_extras(stop_ids),
                    "sampling_seed": seed,
                }
                output = loop.run_until_complete(
                    self._engine.async_generate(
                        prompt=None,
                        sampling_params=per_call_params,
                        return_logprob=True,
                        input_ids=[input_ids],
                        image_data=[image] if image is not None else None,
                    )
                )
                if isinstance(output, list):
                    merged_output.extend(output)
                else:
                    merged_output.append(output)

                if verbose:
                    last_out = merged_output[-1]
                    resp, _ = _post_process_outputs(self.tokenizer, [last_out])
                    meta = last_out.get("meta_info", {}) if isinstance(last_out, dict) else {}
                    plen = len(input_ids)
                    nfe_str = format_nfe_for_log(meta.get("nfe"))
                    log(
                        f"[dream-sglang] sample {j + 1}/{total_batch_size} "
                        f"prompt_tokens={plen} done elapsed={time.time() - t0:.2f}s {nfe_str}"
                    )
                    if sample_meta and j < len(sample_meta):
                        sm = sample_meta[j]
                        log(
                            f"[dream-sglang] sample[{j}] uid={sm.get('uid')} "
                            f"data_source={sm.get('data_source', '?')}"
                        )
        else:
            merged_output = None

        [merged_output] = broadcast_pyobj(
            data=[merged_output],
            rank=self._rank,
            dist_group=self._device_mesh_cpu[self._tp_mesh_name].get_group(),
            src=self._device_mesh_cpu[self._tp_mesh_name].mesh[0].item(),
            force_cpu_device=False,
        )

        finish_reasons = [
            (out.get("meta_info", {}) or {}).get("finish_reason") if isinstance(out, dict) else None
            for out in merged_output
        ]
        response, rollout_log_probs = _post_process_outputs(self.tokenizer, merged_output)
        response, rollout_log_probs = _finalize_dream_response_tensor(
            response,
            rollout_log_probs,
            stop_ids,
            self.pad_token_id,
            self.config.response_length,
            finish_reasons=finish_reasons,
            opening_prefix_ids=self._opening_prefix_ids(),
        )
        response = response.to(idx.device)
        if rollout_log_probs is not None:
            rollout_log_probs = rollout_log_probs.to(idx.device)

        seq = torch.cat([idx_repeat, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids_repeat.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(total_batch_size, 1)
        response_position_ids = position_ids_repeat[:, -1:] + delta_position_id
        position_ids_out = torch.cat([position_ids_repeat, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask_repeat.dtype
        )
        attention_mask_out = torch.cat((attention_mask_repeat, response_attention_mask), dim=-1)

        batch_tensors = {
            "prompts": idx_repeat,
            "responses": response,
            "input_ids": seq,
            "attention_mask": attention_mask_out,
            "position_ids": position_ids_out,
        }
        if rollout_log_probs is not None:
            batch_tensors["rollout_log_probs"] = rollout_log_probs

        batch = TensorDict(batch_tensors, batch_size=total_batch_size)

        if self.config.free_cache_engine and self._engine is not None:
            self._engine.flush_cache()

        if verbose:
            try:
                import torch.distributed as dist

                rank = dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                rank = 0
            log(
                f"[dream-sglang][RANK{rank}] batch_done n_samples={total_batch_size} "
                f"total={time.time() - t_batch:.2f}s"
            )
            log_rollout_batch(
                prompts=prompts,
                responses=response,
                idx_repeat=idx_repeat,
                gen_kwargs={
                    **dict(self.config),
                    "dllm_decode": self.config.get("dllm_decode", "multiblock"),
                    "global_step": prompts.meta_info.get("global_step"),
                },
                tokenizer=self.tokenizer,
                elapsed_s=time.time() - t_batch,
                is_validate=prompts.meta_info.get("validate", False),
                attention_mask=attention_mask_out,
                non_tensor_batch=_non_tensor_batch,
            )

        return DataProto(batch=batch, non_tensor_batch=_non_tensor_batch)
