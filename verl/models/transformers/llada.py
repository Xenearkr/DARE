# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Ulysses sequence parallelism patch for LLaDA models.

LLaDA calls flash_attn_varlen_func directly (not via HuggingFace's _flash_attention_forward),
so the generic monkey patch in monkey_patch.py does not apply. This module provides a patched
attention method for LLaDABlock that inserts all-to-all communication for Ulysses SP.
"""

import sys
from typing import Optional, Tuple

import torch

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads for GQA.
    Input:  (batch, num_kv_heads, seqlen, head_dim)
    Output: (batch, num_kv_heads * n_rep, seqlen, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def llada_ulysses_attention_forward(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
    replace_position: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Patched attention method for LLaDABlock that supports Ulysses sequence parallelism.

    Non-SP path (sp_size == 1 or cu_seqlens is None): identical to original code.
    SP path (sp_size > 1 and cu_seqlens is not None):
      1. Reshape q/k/v from (B, T_local, C) to (B, n_heads, T_local, head_dim)
      2. Handle GQA: repeat_kv on k/v if n_kv_heads != n_heads
      3. All-to-all #1: gather full sequence, scatter heads
      4. Apply RoPE on full sequence (correct positions for all ranks)
      5. Flash attn varlen with cu_seqlens
      6. All-to-all #2: gather heads, scatter sequence back
      7. Reshape back to (B, T_local, C), apply attn_out
    """
    sp_size = get_ulysses_sequence_parallel_world_size()

    B, T, C = q.size()
    dtype = k.dtype

    # Apply layer norms (same as original)
    if self.q_norm is not None and self.k_norm is not None:
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)

    n_heads = self.config.n_heads
    n_kv_heads = self.config.effective_n_kv_heads
    head_dim = C // n_heads

    # Reshape: (B, T, C) -> (B, n_heads, T, head_dim) / (B, n_kv_heads, T, head_dim)
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_kv_heads, head_dim).transpose(1, 2)

    # === Non-SP path: identical to original LLaDABlock.attention ===
    if sp_size <= 1 or cu_seqlens is None:
        if layer_past is not None:
            past_key, past_value = layer_past
            if replace_position is None:
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)
            else:
                B_rp = replace_position.shape[0]
                for batch_idx in range(B_rp):
                    batch_replace_indices = replace_position[batch_idx].nonzero(as_tuple=True)[0]
                    if len(batch_replace_indices) > 0:
                        past_key[batch_idx, :, batch_replace_indices] = k[batch_idx, :, :len(batch_replace_indices)]
                        past_value[batch_idx, :, batch_replace_indices] = v[batch_idx, :, :len(batch_replace_indices)]
                k = past_key
                v = past_value

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]

        if self.config.rope:
            if replace_position is None:
                q, k = self.rotary_emb(q, k)
            else:
                max_replace_pos = replace_position.nonzero(as_tuple=True)[1].max() + 1 if replace_position.any() else key_len
                q, k = self.rotary_emb(q, k, max_replace_pos)

        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        if (
            self.flash_attn_with_kvcache is not None
            and layer_past is not None
            and attention_mask is None
            and replace_position is None
            and cu_seqlens is None
        ):
            att = self.flash_attn_with_kvcache(
                q=q.transpose(1, 2),
                k_cache=k.transpose(1, 2),
                v_cache=v.transpose(1, 2),
                k=None,
                v=None,
                causal=False,
            ).transpose(1, 2)
        else:
            att = self._scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0 if not self.training else self.config.attention_dropout,
                is_causal=False,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_out(att), present

    # === SP path: Ulysses sequence parallelism ===
    assert B == 1, "Sequence parallelism requires batch_size == 1 (packed sequences)"
    present = None  # No KV cache in SP mode

    # Handle GQA: repeat k/v to match n_heads for uniform all-to-all
    if n_kv_heads != n_heads:
        n_rep = n_heads // n_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

    # All-to-all #1: gather full sequence, scatter heads
    # (1, n_heads, T_local, head_dim) -> (1, n_heads/sp, T_full, head_dim)
    total_nnz = cu_seqlens[-1].item()
    q = gather_seq_scatter_heads(q, seq_dim=2, head_dim=1, unpadded_dim_size=total_nnz)
    k = gather_seq_scatter_heads(k, seq_dim=2, head_dim=1, unpadded_dim_size=total_nnz)
    v = gather_seq_scatter_heads(v, seq_dim=2, head_dim=1, unpadded_dim_size=total_nnz)

    # Apply RoPE on FULL sequence (positions [0..T_full-1] are correct for all ranks)
    if self.config.rope:
        q, k = self.rotary_emb(q, k)

    # Flash attn varlen
    # (1, n_heads/sp, T_full, head_dim) -> (T_full, n_heads/sp, head_dim)
    q_flat = q.squeeze(0).permute(1, 0, 2).contiguous()
    k_flat = k.squeeze(0).permute(1, 0, 2).contiguous()
    v_flat = v.squeeze(0).permute(1, 0, 2).contiguous()

    dropout_p = 0.0 if not self.training else self.config.attention_dropout
    att = self.flash_attn_varlen_func(
        q_flat, k_flat, v_flat,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout_p,
        causal=False,
    )  # (T_full, n_heads/sp, head_dim)

    # Reshape back: (T_full, n_heads/sp, head_dim) -> (1, n_heads/sp, T_full, head_dim)
    att = att.unsqueeze(0).transpose(1, 2)

    # All-to-all #2: gather heads, scatter sequence
    # (1, n_heads/sp, T_full, head_dim) -> (1, n_heads, T_local, head_dim)
    att = gather_heads_scatter_seq(att, head_dim=1, seq_dim=2)

    # Reshape back to (B, T, C) and apply output projection
    att = att.transpose(1, 2).contiguous().view(B, T, C)
    return self.attn_out(att), present


def apply_llada_ulysses_patch(model):
    """Patch LLaDABlock.attention with Ulysses SP-aware version."""
    module = sys.modules[model.__module__]
    module.LLaDABlock.attention = llada_ulysses_attention_forward
    print("Monkey patch LLaDABlock.attention for LLaDA Ulysses SP")
