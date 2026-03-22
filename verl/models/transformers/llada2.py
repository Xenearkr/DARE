import sys
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import MoeModelOutputWithPast

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
)


def _use_llada2_ulysses_sp(attention_mask: Optional[torch.Tensor], local_seq_len: int) -> bool:
    if get_ulysses_sequence_parallel_world_size() <= 1:
        return False
    if attention_mask is None or attention_mask.dim() != 4:
        return False
    return attention_mask.size(-1) != local_seq_len


def llada2_ulysses_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    if not _use_llada2_ulysses_sp(attention_mask, hidden_states.size(1)):
        return self.__class__._original_ulysses_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    if past_key_value is not None or use_cache:
        raise NotImplementedError("LLaDA2 Ulysses sequence parallelism does not support KV cache.")
    if output_attentions:
        raise NotImplementedError("LLaDA2 Ulysses sequence parallelism does not return attention weights.")
    if position_embeddings is None:
        raise ValueError("position_embeddings is required for LLaDA2 Ulysses sequence parallelism.")

    input_shape = hidden_states.shape[:-1]
    bsz, q_len, _ = hidden_states.size()

    qkv = self.query_key_value(hidden_states)
    qkv = qkv.view(
        bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim
    )

    query_states, key_states, value_states = qkv.split(
        [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
    )
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if self.config.use_qk_norm:
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

    cos, sin = position_embeddings
    query_states, key_states = sys.modules[self.__module__].apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    # Match eager attention semantics before the all-to-all step.
    key_states = sys.modules[self.__module__].repeat_kv(key_states, self.num_key_value_groups)
    value_states = sys.modules[self.__module__].repeat_kv(value_states, self.num_key_value_groups)

    global_seq_len = attention_mask.size(-1)
    query_states = gather_seq_scatter_heads(
        query_states, seq_dim=2, head_dim=1, unpadded_dim_size=global_seq_len
    )
    key_states = gather_seq_scatter_heads(
        key_states, seq_dim=2, head_dim=1, unpadded_dim_size=global_seq_len
    )
    value_states = gather_seq_scatter_heads(
        value_states, seq_dim=2, head_dim=1, unpadded_dim_size=global_seq_len
    )

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    attn_weights = attn_weights + attention_mask[:, :, :global_seq_len, :global_seq_len]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training
    )

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=2)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.dense(attn_output)

    return attn_output, None, None


def llada2_ulysses_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, MoeModelOutputWithPast]:
    local_seq_len = 0
    if input_ids is not None:
        local_seq_len = input_ids.shape[1]
    elif inputs_embeds is not None:
        local_seq_len = inputs_embeds.shape[1]

    if not _use_llada2_ulysses_sp(attention_mask, local_seq_len):
        return self.__class__._original_ulysses_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            **kwargs,
        )

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    if input_ids is not None:
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    if position_ids is None:
        raise ValueError("LLaDA2 Ulysses sequence parallelism requires sliced position_ids.")
    if attention_mask is None or attention_mask.dim() != 4:
        raise ValueError("LLaDA2 Ulysses sequence parallelism requires a full 4D block attention mask.")

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_router_logits = () if output_router_logits else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                output_router_logits,
                use_cache,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
        if output_router_logits and layer_outputs[-1] is not None:
            all_router_logits += (layer_outputs[-1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                all_router_logits,
            ]
            if v is not None
        )

    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=all_router_logits,
    )


def apply_llada2_ulysses_patch(model):
    module = sys.modules[model.__module__]

    attention_classes = [module.LLaDA2MoeAttention]
    if hasattr(module, "LLaDA2MoeSdpaAttention"):
        attention_classes.append(module.LLaDA2MoeSdpaAttention)
    if hasattr(module, "LLaDA2MoeFlexAttention"):
        attention_classes.append(module.LLaDA2MoeFlexAttention)

    for attention_cls in attention_classes:
        if not hasattr(attention_cls, "_original_ulysses_forward"):
            attention_cls._original_ulysses_forward = attention_cls.forward
    if not hasattr(module.LLaDA2MoeModel, "_original_ulysses_forward"):
        module.LLaDA2MoeModel._original_ulysses_forward = module.LLaDA2MoeModel.forward

    for attention_cls in attention_classes:
        attention_cls.forward = llada2_ulysses_attention_forward
    module.LLaDA2MoeModel.forward = llada2_ulysses_model_forward
    print("Monkey patch LLaDA2Moe attention/model for Ulysses SP")
