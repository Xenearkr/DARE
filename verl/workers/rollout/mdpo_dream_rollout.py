# Copyright 2025 Shanghai AI Lab
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
"""
MDPO-specific rollout for Dream that collects the full diffusion trajectory.
Unlike standard rollout which only returns final sequences, this collects
input/output/confidence at every denoising step for step-wise reward computation.

Uses batchified (non-packed) generation following the MDPO source code style,
adapted for Dream's logit-shifting and timestep-based unmasking schedule.
"""

import torch
import torch.nn.functional as F
import torch.distributions as dists
import torch.distributed as dist
import numpy as np
from tensordict import TensorDict
from torch import nn
import time

from verl import DataProto
from verl.workers.rollout.base import BaseRollout


__all__ = ["MDPORollout"]


def top_p_logits(logits, top_p=None):
    """Nucleus (top-p) filtering from MDPO source."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    """Top-k filtering from MDPO source."""
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


@torch.no_grad()
def mdpo_diffusion_generate_dream(model, prompt, mask_id, prompt_mask=None,
                                    steps=128, gen_length=128, temperature=0.,
                                    do_sample=True, conf_alg='entropy',
                                    top_p=None, top_k=None):
    """
    MDPO diffusion generation for Dream, closely following the MDPO source code style.
    Uses standard HuggingFace model call with attention_mask (no packed sequences).

    Dream-specific differences from LLaDA:
      - Logit shift: logits = cat([logits[:, :1], logits[:, :-1]], dim=1)
      - Timestep-based unmasking schedule instead of block-based get_num_transfer_tokens

    Args:
        model: The Dream model.
        prompt: (batch_size, prompt_length) input prompt tokens.
        mask_id: Mask token ID.
        prompt_mask: (batch_size, prompt_length) attention mask for prompt. None = all ones.
        steps: Number of diffusion steps.
        gen_length: Length of generation per sequence.
        temperature: Sampling temperature.
        do_sample: Whether to sample from the distribution.
        conf_alg: Confidence algorithm ("random", "entropy", "topk_margin", or default).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling threshold.

    Returns:
        final_response: (batch_size, gen_length) Final generated response tokens.
        intermediate_results: List of (batch_size, gen_length) tensors - predicted x0 at each step.
        intermediate_confidence: List of (batch_size, gen_length) tensors - confidence at each step.
        intermediate_inputs: List of (batch_size, gen_length) tensors - input state at each step.
    """
    batch_size = prompt.shape[0]
    prompt_length = prompt.shape[1]
    max_length = prompt_length + gen_length
    eps = 1e-3

    if prompt_mask is None:
        prompt_mask = torch.ones_like(prompt)

    with torch.amp.autocast("cuda", enabled=True):
        # Initialize: [prompt | mask_tokens]
        x = F.pad(prompt, (0, gen_length), value=mask_id)

        if torch.any(prompt_mask == 0.0):
            attn_mask = F.pad(prompt_mask, (0, gen_length), value=1.0)
        else:
            attn_mask = torch.ones(batch_size, max_length,
                                    device=prompt.device, dtype=prompt_mask.dtype)

        # Dream uses a linear timestep schedule for unmasking
        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        intermediate_inputs = []
        intermediate_results = []
        intermediate_confidence = []

        for i in range(steps):
            mask_index = (x == mask_id)

            # Record input state (response region only)
            intermediate_inputs.append(x[:, -gen_length:].clone())

            # Model forward – standard HuggingFace call
            logits = model(x, attention_mask=attn_mask).logits
            # Dream-specific: shift logits by 1 position
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-p / top-k filtering
            if top_p is not None and top_p < 1:
                logits = top_p_logits(logits, top_p)
            if top_k is not None and top_k > 0:
                logits = top_k_logits(logits, top_k)

            # Compute softmax probabilities
            p = F.softmax(logits, dim=-1)

            # Sample tokens
            if do_sample and temperature > 0:
                try:
                    x0 = dists.Categorical(probs=p).sample()
                    confidence = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                except Exception:
                    confidence, x0 = p.max(dim=-1)
            else:
                confidence, x0 = p.max(dim=-1)

            # Save token-level confidence for MDPO training
            intermediate_confidence.append(confidence[:, -gen_length:].clone())

            # Determine remasking strategy
            if conf_alg == 'random':
                remasking_confidence = torch.rand(x.shape, device=x.device)
            elif conf_alg == 'entropy':
                epsilon_val = 1e-10
                log_probs = torch.log(p + epsilon_val)
                remasking_confidence = torch.sum(p * log_probs, dim=-1)
            elif conf_alg == 'topk_margin':
                sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                top1_probs = sorted_probs[:, :, 0]
                top2_probs = sorted_probs[:, :, 1]
                remasking_confidence = top1_probs - top2_probs
            else:
                # default: use token confidence for remasking
                remasking_confidence = confidence.clone()

            # Only update masked positions
            x0 = torch.where(mask_index, x0, x)
            intermediate_results.append(x0[:, -gen_length:].clone())

            remasking_confidence = torch.where(
                mask_index, remasking_confidence,
                torch.tensor(-np.inf, device=x.device),
            )

            # Dream timestep-based unmasking: determine how many tokens to unmask
            t = timesteps[i]
            s = timesteps[i + 1]
            target_mask_count = (mask_index.sum(dim=-1).float() * (s / t)).long()

            for b in range(batch_size):
                current_mask = mask_index[b]
                n_masked = current_mask.sum().item()
                n_to_unmask = n_masked - target_mask_count[b].item()
                n_to_unmask = max(0, min(n_to_unmask, n_masked))

                if n_to_unmask > 0:
                    # Select highest remasking-confidence masked tokens to unmask
                    masked_conf = remasking_confidence[b].clone()
                    masked_conf[~current_mask] = -np.inf
                    _, select_indices = torch.topk(masked_conf, k=n_to_unmask)
                    x[b, select_indices] = x0[b, select_indices]

    return x[:, -gen_length:], intermediate_results, intermediate_confidence, intermediate_inputs


class MDPORollout(BaseRollout):
    def __init__(self, module: nn.Module, config, tokenizer):
        """MDPO rollout for Dream that collects full diffusion trajectories."""
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = tokenizer
        self.MASK_TOKEN_ID = self.module.config.mask_token_id
        self.PAD_TOKEN_ID = self.module.config.pad_token_id
        self.EOS_TOKEN_ID = self.module.config.eos_token_id

        # diffusion related parameters
        self.response_length = config["response_length"]
        self.num_diffusion_steps = config["num_diffusion_steps"]
        self.block_length = config["block_length"]
        self.cfg_scale = config["cfg_scale"]

        # MDPO specific parameters
        self.conf_alg = config.get("conf_alg", "entropy")
        self.rcr = config.get("rcr", False)
        self.top_p = config.get("top_p", None)
        self.top_k = config.get("top_k", None)

        # rollout related parameters
        self.n_rollout = config["n"]
        self.temperature = config["temperature"]
        self.do_sample = config["do_sample"]
        self.val_kwargs = config["val_kwargs"]

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using MDPO diffusion process for Dream, collecting the full trajectory.

        Returns DataProto containing:
            - responses: final generated completion (batch, response_len)
            - all_steps_input_ids: input at each diffusion step (batch, steps, response_len)
            - all_steps_completion_ids: output at each diffusion step (batch, steps, response_len)
            - all_confidence: confidence scores at each step (batch, steps, response_len)
        """
        start_time = time.time()

        idx = prompts.batch["input_ids"]           # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention mask
        position_ids = prompts.batch["position_ids"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()

        is_validate = prompts.meta_info.get("validate", False)

        n_rollout = 1 if is_validate else self.n_rollout
        num_diffusion_steps = self.val_kwargs["num_diffusion_steps"] if is_validate else self.num_diffusion_steps
        temperature = self.val_kwargs.get("temperature", self.temperature) if is_validate else self.temperature
        do_sample = self.val_kwargs.get("do_sample", self.do_sample) if is_validate else self.do_sample

        idx_repeat = idx.repeat_interleave(n_rollout, dim=0)
        attention_mask_repeat = attention_mask.repeat_interleave(n_rollout, dim=0)
        total_batch_size = batch_size * n_rollout

        # ---------- batchified generation (one sample at a time, following MDPO source) ----------
        all_responses = []
        all_intermediate_inputs = []
        all_intermediate_results = []
        all_intermediate_confidence = []

        for sample_idx in range(total_batch_size):
            prompt_i = idx_repeat[sample_idx:sample_idx + 1]          # (1, prompt_length)
            prompt_mask_i = attention_mask_repeat[sample_idx:sample_idx + 1]  # (1, prompt_length)

            final_resp, inter_results, inter_conf, inter_inputs = mdpo_diffusion_generate_dream(
                model=self.module,
                prompt=prompt_i,
                mask_id=self.MASK_TOKEN_ID,
                prompt_mask=prompt_mask_i,
                steps=num_diffusion_steps,
                gen_length=self.response_length,
                temperature=temperature,
                do_sample=do_sample,
                conf_alg=self.conf_alg,
                top_p=self.top_p,
                top_k=self.top_k,
            )

            # final_resp: (1, response_len)
            all_responses.append(final_resp.squeeze(0))

            # inter_inputs / inter_results / inter_conf: list of (1, response_len)
            all_intermediate_inputs.append(torch.stack([t.squeeze(0) for t in inter_inputs]))
            all_intermediate_results.append(torch.stack([t.squeeze(0) for t in inter_results]))
            all_intermediate_confidence.append(torch.stack([t.squeeze(0) for t in inter_conf]))

            answer = self.tokenizer.decode(final_resp.squeeze(0), skip_special_tokens=True)
            if not is_validate:
                print(f"==================[RANK{dist.get_rank()}] rollout question ID: {sample_idx}=================\n"
                      f"Generated answer: {answer}\n==========================================")
            else:
                print(f"==================[RANK{dist.get_rank()}] validation question ID: {sample_idx}=================\n"
                      f"Generated answer: {answer}\n==========================================")

        # ---------- assemble output ----------
        responses = torch.stack(all_responses)  # (total_batch_size, response_len)

        # Build attention masks: prompt attn + response attn (mark EOS-and-after as 0)
        resp_attention = torch.ones(total_batch_size, self.response_length,
                                     device=idx.device, dtype=attention_mask.dtype)
        for b in range(total_batch_size):
            eos_positions = (responses[b] == self.EOS_TOKEN_ID).nonzero()
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                resp_attention[b, first_eos + 1:] = 0
        full_attention_mask = torch.cat([attention_mask_repeat, resp_attention], dim=1)

        input_ids = torch.cat([idx_repeat, responses], dim=1)

        # Stack trajectory data: (batch, steps, response_len)
        all_steps_input_ids = torch.stack(all_intermediate_inputs)           # (batch, steps, resp_len)
        all_steps_completion_ids = torch.stack(all_intermediate_results)     # (batch, steps, resp_len)
        all_confidence_tensor = torch.stack(all_intermediate_confidence)     # (batch, steps, resp_len)

        num_steps = all_steps_input_ids.shape[1]

        batch = TensorDict(
            {
                "prompts": idx_repeat,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": full_attention_mask,
                "position_ids": torch.cat([
                    position_ids.repeat_interleave(n_rollout, dim=0),
                    position_ids[:, -1:].repeat_interleave(n_rollout, dim=0) +
                    torch.arange(1, self.response_length + 1, device=position_ids.device)
                ], dim=1),
                "all_steps_input_ids": all_steps_input_ids,
                "all_steps_completion_ids": all_steps_completion_ids,
                "all_confidence": all_confidence_tensor,
            },
            batch_size=total_batch_size,
        )

        self.module.train()

        total_time = time.time() - start_time
        print(f"[RANK{dist.get_rank()}] MDPO Dream generate_sequences total time: {total_time:.2f}s, "
              f"collected {num_steps} diffusion steps for {total_batch_size} samples")

        result = DataProto(batch=batch)
        result.meta_info["mask_token_id"] = self.MASK_TOKEN_ID
        return result
