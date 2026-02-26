import os
import json
import random
import argparse
import re
import sys
import time
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

seed = 42
random.seed(seed)

def add_gumbel_noise(logits, temperature):
    """Add Gumbel noise to logits for sampling"""
    if temperature == 0.:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)  # Generate uniform noise with same shape as logits
    gumbel_noise = (- torch.log(noise)) ** temperature  # Calculate Gumbel noise
    return logits.exp() / gumbel_noise  # Return logits with added noise


def get_num_transfer_tokens(mask_index, steps):
    """Calculate the number of tokens to fill in each step"""
    mask_num = mask_index.sum(dim=1, keepdim=True)  # Count the number of masks in each sample
    base = mask_num // steps  # Average allocated to each step
    remainder = mask_num % steps  # Calculate remainder
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:  # If there is a remainder
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1  # Allocate remainder
    return num_transfer_tokens.to(torch.int64)  # Return the number of tokens allocated to each step (batch_size, steps)


@torch.no_grad()
def generate(model, packed_sequence, cu_seqlens, max_seqlen, steps=128, gen_length=128, block_length=128, temperature=0.0, cfg_scale=0.0, remasking="low_confidence", mask_id=126336, eos_id=126081, mode="eval"):
    """
    Optimized version of the generate function that works directly with packed sequences.
    
    Args:
        packed_sequence: Tensor of shape (1, (prompt_len1+gen_length) + (prompt_len2+gen_length) + ... + (prompt_lenN+gen_length)) containing the packed input
        cu_seqlens: Cumulative sequence lengths tensor
        max_seqlen: Maximum sequence length
        steps: Number of generation steps
        gen_length: Length of generation
        block_length: Length of each generation block
        temperature: Temperature for sampling
        cfg_scale: Classifier-free guidance scale
        remasking: Remasking strategy
        mask_id: Mask token ID
    """
    batch_size = len(cu_seqlens) - 1
    # print(f"cu_seqlens: {cu_seqlens}")
    
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Initialize x with mask tokens for generation part
        x = packed_sequence.clone()
        
        # Calculate prompt lengths for each sequence
        prompt_lengths = []
        prompt_index = torch.zeros_like(x, dtype=torch.bool)
        for j in range(batch_size):
            prompt_len = cu_seqlens[j + 1] - cu_seqlens[j] - gen_length
            prompt_lengths.append(prompt_len)
            prompt_index[0, cu_seqlens[j]:cu_seqlens[j] + prompt_len] = True

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in range(num_blocks):
            start_block = num_block * block_length
            end_block = (num_block + 1) * block_length
            
            # Calculate block mask for the entire packed sequence
            num_transfer_tokens = torch.zeros(batch_size, steps_per_block, device=x.device, dtype=torch.int64)
            for j in range(batch_size):
                seq_start = cu_seqlens[j] + prompt_lengths[j] + start_block
                seq_end = cu_seqlens[j] + prompt_lengths[j] + end_block
                num_transfer_tokens[j] = get_num_transfer_tokens((x[0, seq_start:seq_end] == mask_id).unsqueeze(0), steps_per_block)
            
            # Start diffusion for the current block. The number of tokens generated in each step is basically the same.
            for i in range(steps_per_block):
                # print(f"block {num_block}, step {i}")
                mask_index = (x == mask_id)

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    
                    # Concatenate packed sequences
                    x_combined = torch.cat([x, un_x], dim=0)
                    combined_cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[1:] + cu_seqlens[-1]], dim=0)
                    
                    # Get logits in a single forward pass
                    logits = model(x_combined, cu_seqlens=combined_cu_seqlens, max_seqlen=max_seqlen).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).logits
                    # # for j in range(batch_size):
                    # j=1
                    # print(f"logits[{j}] sum: {logits[0, cu_seqlens[j]:cu_seqlens[j + 1]].sum()}")
                
                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1).to(x.device)  # New sequence after generating a block
                # print(f"logits_with_noise[1] sum: {logits_with_noise[0, cu_seqlens[1]:cu_seqlens[2]].sum()}")
                # print(f"x0[1] sum: {x0[0, cu_seqlens[1]:cu_seqlens[2]].sum()}")

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1).to(x.device)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # Probability of the current generated token (x0)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)  # Replace the newly generated token (x0) with the mask position
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))  # Set confidence to -inf for non-mask positions to ensure they are not updated

                # Set confidence of EOS token to 0
                if mode == "eval":
                    confidence[x0 == eos_id] = 0.0

                # Select tokens to transfer based on confidence for each sequence
                for j in range(batch_size):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        # Get confidence scores for the current sequence's generation part
                        seq_start = cu_seqlens[j] + prompt_lengths[j] + start_block
                        seq_end = cu_seqlens[j] + prompt_lengths[j] + end_block
                        seq_confidence = confidence[0, seq_start:seq_end]
                        
                        # Get top-k indices for this sequence
                        _, select_indices = torch.topk(seq_confidence, k=num_tokens)
                        
                        # Update the selected tokens
                        if len(select_indices) > 0:
                            global_indices = seq_start + select_indices
                            x[0, global_indices] = x0[0, global_indices]
        
        # Replace remaining mask tokens with eos token
        x[x == mask_id] = eos_id
        return x


def get_num_transfer_tokens_fastdllm(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)
        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index


def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


@torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, attention_mask=None, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_fastdllm(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1

    return x


@torch.no_grad()
@torch.compile(mode="max-autotune", fullgraph=True)
def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, attention_mask=None, threshold=None, factor=None
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # shape expected by get_transfer_index*

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation

    return x


@torch.no_grad()
def reversed_process_llada(
    model,
    packed_sequence,
    cu_seqlens, 
    max_seqlen,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    eos_id=126081,
    mode="eval",
    step_merge=False,
):
    batch_size = len(cu_seqlens) - 1

    # Initialize x with mask tokens for generation part
    x = packed_sequence.clone()

    # Calculate prompt lengths for each sequence
    prompt_lengths = []
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    for j in range(batch_size):
        prompt_len = cu_seqlens[j + 1] - cu_seqlens[j] - gen_length
        prompt_lengths.append(prompt_len)
        prompt_index[0, cu_seqlens[j]:cu_seqlens[j] + prompt_len] = True

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)
    ### Record the trajectory changes of x ###
    if not step_merge:
        reversed_traj_unmask_positions = []
    if step_merge:
        mereged_reversed_traj_unmask_positions = []

    for num_block in range(num_blocks):
        start_block = num_block * block_length
        end_block = (num_block + 1) * block_length
        mask_index_per_block = (x == mask_id)

        # Calculate block mask for the entire packed sequence
        num_transfer_tokens = torch.zeros(batch_size, steps_per_block, device=x.device, dtype=torch.int64)
        for j in range(batch_size):
            seq_start = cu_seqlens[j] + prompt_lengths[j] + start_block
            seq_end = cu_seqlens[j] + prompt_lengths[j] + end_block
            num_transfer_tokens[j] = get_num_transfer_tokens((x[0, seq_start:seq_end] == mask_id).unsqueeze(0), steps_per_block)

        # Start diffusion for the current block. The number of tokens generated in each step is basically the same.
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            # Handle classifier-free guidance more efficiently
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_combined = torch.cat([x, un_x], dim=0)
                combined_cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[1:] + cu_seqlens[-1]], dim=0)
                # Get logits in a single forward pass
                logits = model(x_combined, cu_seqlens=combined_cu_seqlens, max_seqlen=max_seqlen).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).logits
            # Apply Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            del logits_with_noise
            # Handle remasking strategy
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Update masked tokens
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))
            # Set confidence of EOS token to 0
            if mode == "eval":
                confidence[x0 == eos_id] = 0.0

            # Select tokens to transfer based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(batch_size):
                num_tokens = num_transfer_tokens[j, i].item()
                if num_tokens > 0:
                    # Get confidence scores for the current sequence's generation part
                    seq_start = cu_seqlens[j] + prompt_lengths[j] + start_block
                    seq_end = cu_seqlens[j] + prompt_lengths[j] + end_block
                    seq_confidence = confidence[0, seq_start:seq_end]
                    
                    # Get top-k indices for this sequence
                    _, select_indices = torch.topk(seq_confidence, k=num_tokens)
                    # Update the selected tokens
                    if len(select_indices) > 0:
                        global_indices = seq_start + select_indices
                        x[0, global_indices] = x0[0, global_indices]

            x[transfer_index] = x0[transfer_index]
            if not step_merge:
                unmask_positions = (mask_index & (x != mask_id))
                reversed_traj_unmask_positions.append(unmask_positions)
            # if step merge, merge intermediate states and save the last state in a block
            if step_merge and i == (steps_per_block - 1):
                unmask_positions = (mask_index_per_block & (x != mask_id))
                mereged_reversed_traj_unmask_positions.append(unmask_positions)

    # Replace remaining mask tokens with eos token
    x[x == mask_id] = eos_id
    if step_merge:
        return x, torch.stack(mereged_reversed_traj_unmask_positions, dim=1)
    else:
        return x, torch.stack(reversed_traj_unmask_positions, dim=1)


@torch.no_grad()
def fast_reversed_process_llada(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    attention_mask=None,
    step_merge=False,
    threshold=None, 
    factor=None,
):
    bs = prompt.shape[0]
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    ### Record the trajectory changes of x ###
    if not step_merge:
        reversed_traj_unmask_positions = []
    else:
        merged_reversed_traj_unmask_positions = []
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        mask_index_per_block = (x == mask_id)
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_fastdllm(block_mask_index, steps)
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        if not step_merge:
            unmask_positions = (mask_index & (x != mask_id))
            reversed_traj_unmask_positions.append(unmask_positions)

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values

        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            full_mask_index = (x == mask_id)
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            if not step_merge:
                unmask_positions = (full_mask_index & (x != mask_id))
                reversed_traj_unmask_positions.append(unmask_positions)
            elif step_merge and i == (steps_per_block - 1):
                merged_unmask_positions = (mask_index_per_block & (x != mask_id))
                merged_reversed_traj_unmask_positions.append(merged_unmask_positions)

            i += 1
        
    if step_merge:
        return x, torch.stack(merged_reversed_traj_unmask_positions, dim=1)
    else:
        return x, torch.stack(reversed_traj_unmask_positions, dim=1)


@torch.no_grad()
def fast_reversed_process_dual_cache_llada(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    attention_mask=None,
    step_merge=True,
    threshold=None, 
    factor=None,
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    ### Record the trajectory changes of x ###
    if not step_merge:
        reversed_traj_unmask_positions = []
    if step_merge:
        mereged_reversed_traj_unmask_positions = []

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        mask_index_per_block = (x == mask_id)
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False
        mask_index_per_block[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )
        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # shape expected by get_transfer_index*

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)

            mask_index_per_block = (x == mask_id)
            mask_index_per_block[:, e:] = False

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation
 
            
        if not step_merge:
            unmask_positions = (global_mask_index & (x != mask_id))[:, -gen_length:]
            reversed_traj_unmask_positions.append(unmask_positions)
        # if step merge, merge intermediate states and save the last state in a block
        if step_merge and i == (steps - 1):
            unmask_positions = (mask_index_per_block & (x != mask_id))[:, -gen_length:]
            mereged_reversed_traj_unmask_positions.append(unmask_positions)
        
    if step_merge:
        return x, torch.stack(mereged_reversed_traj_unmask_positions, dim=1)
    else:
        return x, torch.stack(reversed_traj_unmask_positions, dim=1)