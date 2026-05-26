"""
Common utilities for rollout operations, including packing and generation functions.
"""
import re
import torch
import torch.distributed as dist
import time
from typing import List, Dict, Any, Tuple
from .generate import generate, generate_with_prefix_cache, generate_with_dual_cache, reversed_process_llada, fast_reversed_process_llada, fast_reversed_process_dual_cache_llada

### pack sequences ###
def pack_sequences(
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    mask_token_id: int,
    max_model_length: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Pre-pack all data to get all packed batches on this rank
    
    Args:
        idx_repeat: Repeated input sequences (total_batch_size, prompt_length)
        attention_mask_repeat: Repeated attention masks (total_batch_size, prompt_length)
        response_length: Response length
        mask_token_id: Mask token id
        max_model_length: Maximum length of a pack data
        device: Device
        
    Returns:
        packs: List containing all pack information
    """
    packs = []
    total_batch_size = idx_repeat.size(0)
    i = 0
    
    while i < total_batch_size:
        full_input_ids_list = []
        prompt_lengths = []
        current_length = 0
        cu_seqlens = [0]
        max_seqlen = 0
        
        while i < total_batch_size:
            input_ids = idx_repeat[i, attention_mask_repeat[i] == 1].tolist()  # Only take valid tokens
            full_input_ids = input_ids + [mask_token_id] * response_length
            if current_length + len(full_input_ids) > max_model_length:
                break
            full_input_ids_tensor = torch.tensor(full_input_ids, device=device, dtype=torch.long)
            full_input_ids_list.append(full_input_ids_tensor)
            prompt_lengths.append(len(input_ids))
            current_length += len(full_input_ids)
            cu_seqlens.append(current_length)
            max_seqlen = max(max_seqlen, len(full_input_ids))
            i += 1
            
        if not full_input_ids_list:
            raise ValueError("packed failed!")

        packed_input = torch.cat(full_input_ids_list, dim=0).unsqueeze(0)  # (1, total_len)
        packs.append({
            "packed_input": packed_input,
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
            "max_seqlen": max_seqlen,
            "batch_start_idx": i - len(full_input_ids_list),  # The start index of the current pack in idx_repeat
            "num_sequences": len(full_input_ids_list),
            "prompt_lengths": prompt_lengths,
            "is_dummy": False,
        })
    
    return packs


def align_pack_counts(
    packs: List[Dict[str, Any]], 
    prompt_length: int, 
    response_length: int, 
    pad_token_id: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Align the number of pack with other ranks, using dummy packs to fill the difference
    
    Args:
        packs: Local packs list
        prompt_length: Prompt length
        response_length: Response length
        pad_token_id: Pad token id
        device: Device
        
    Returns:
        aligned_packs: Aligned packs list
    """
    local_count = len(packs)  # Local packs count
    
    try:
        world_size = dist.get_world_size()
    except Exception:
        world_size = 1
        
    if world_size > 1:
        local_count_tensor = torch.tensor([local_count], device=device, dtype=torch.int64)
        gathered = [torch.zeros_like(local_count_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_count_tensor)
        max_count = int(torch.stack(gathered, dim=0).max().item())
    else:
        max_count = local_count

    # Use empty data to fill to the same number
    aligned_packs = packs.copy()
    for _ in range(max_count - local_count):
        dummy_len = prompt_length + response_length
        dummy_input = torch.full((1, dummy_len), fill_value=pad_token_id, device=device, dtype=torch.long)
        aligned_packs.append({
            "packed_input": dummy_input,
            "cu_seqlens": torch.tensor([0, dummy_len], dtype=torch.int32, device=device),
            "max_seqlen": dummy_len,
            "batch_start_idx": -1,
            "num_sequences": 1,
            "prompt_lengths": [prompt_length],
            "is_dummy": True,
        })
    
    return aligned_packs


### llada rollout utils ###
def process_llada_generation_outputs(
    pack: Dict[str, Any],
    outputs: torch.Tensor,
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Process the generation outputs, extract responses, attention_masks and answers
    
    Args:
        pack: Pack info dict
        outputs: Model generation outputs
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer
        device: Device
        
    Returns:
        batch_responses: Batch responses (batch, response_length) or None (if dummy)
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    if pack["is_dummy"]:
        # directly create placeholder for dummy data
        batch_full_input_ids = torch.full((1, pack["prompt_lengths"][0] + response_length), fill_value=tokenizer.pad_token_id, device=device, dtype=torch.long)
        batch_attention_masks = torch.ones((1, pack["prompt_lengths"][0] + response_length), device=device, dtype=torch.long)
        return None, batch_full_input_ids, batch_attention_masks, [["dummy"]]

    # Process real data
    batch_responses = []
    batch_answers = []
    batch_attention_masks = []
    cu_seqlens = pack["cu_seqlens"].tolist()
    batch_start_idx = pack["batch_start_idx"]
    prompt_lengths = pack["prompt_lengths"]
    
    for j in range(pack["num_sequences"]):
        seq_start = cu_seqlens[j]
        valid_prompt_len = prompt_lengths[j]
        response = outputs[0, seq_start + valid_prompt_len:seq_start + valid_prompt_len + response_length].unsqueeze(0)  # (1, response_length)
        response_mask = torch.ones((1, response_length), device=device)  # response part is all 1
        batch_responses.append(response)
        batch_attention_masks.append(torch.cat([attention_mask_repeat[batch_start_idx + j].unsqueeze(0), response_mask], dim=1).long())  # (1, seq_length)
        
        # Extract answers
        response_str = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
        boxed_matches = re.findall(r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}", response_str, re.DOTALL)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
        
        answers = list(set(boxed_matches + answer_matches))  # Merge and deduplicate
        batch_answers.append(answers)

    batch_responses = torch.cat(batch_responses, dim=0)  # (batch, response_length)
    batch_full_input_ids = torch.cat([idx_repeat[batch_start_idx:batch_start_idx + pack["num_sequences"]], batch_responses], dim=1)  # (batch, seq_len)
    batch_attention_masks = torch.cat(batch_attention_masks, dim=0)
    
    return batch_responses, batch_full_input_ids, batch_attention_masks, batch_answers


def execute_llada_generation(
    pack: Dict[str, Any],
    module: torch.nn.Module,
    gen_kwargs: Dict[str, Any],
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Execute the generation of a single pack, return the generation results
    
    Args:
        pack: Pack info dict
        module: Model
        gen_kwargs: Generation parameters
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer
        
    Returns:
        batch_responses: Batch responses (batch, response_length) or None (if dummy)
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    batch_start_time = time.time()
    outputs = generate(
        module,
        pack["packed_input"],
        cu_seqlens=pack["cu_seqlens"],
        max_seqlen=pack["max_seqlen"],
        **gen_kwargs,
    )
    print(f"[RANK{dist.get_rank()}] {pack['batch_start_idx']}~{pack['batch_start_idx']+pack['num_sequences']} samples cost: {(time.time() - batch_start_time):.2f}s")

    # Process generation outputs
    batch_responses, batch_full_input_ids, batch_attention_masks, batch_answers = process_llada_generation_outputs(
        pack, outputs, idx_repeat, attention_mask_repeat, response_length, tokenizer, module.device
    )
    
    return batch_responses, batch_full_input_ids, batch_attention_masks, batch_answers


def process_fastllada_generation_outputs(
    outputs: torch.Tensor,
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    # Process real data
    batch_size = outputs.size(0)
    prompt_length = idx_repeat.size(1)
    
    # 1. Extract Responses
    # outputs structure is [Prompt, Response]
    # We slice from prompt_length to the end to get responses
    batch_responses = outputs[:, prompt_length:]

    # 2. Full Input IDs
    # The outputs from generate_with_prefix_cache are already the full sequences
    batch_full_input_ids = outputs

    # 3. Attention Masks
    # Concatenate original prompt mask with all-ones mask for response
    # Response part is fully generated, so it's all valid (1)
    response_mask = torch.ones((batch_size, response_length), device=device, dtype=attention_mask_repeat.dtype)
    batch_attention_masks = torch.cat([attention_mask_repeat, response_mask], dim=1)

    # 4. Extract Answers
    batch_answers = []
    
    # Use batch_decode for efficiency
    decoded_responses = tokenizer.batch_decode(batch_responses, skip_special_tokens=True)
    for response_str in decoded_responses:
        boxed_matches = re.findall(r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}", response_str, re.DOTALL)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
        
        answers = list(set(boxed_matches + answer_matches))  # Merge and deduplicate
        batch_answers.append(answers)
    
    
    return batch_responses, batch_full_input_ids, batch_attention_masks, batch_answers


def execute_fastllada_generation(
    module,
    gen_kwargs: Dict[str, Any],
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Execute the generation of a single pack, return the generation results
    
    Args:
        pack: Pack info dict
        module: Model
        gen_kwargs: Generation parameters
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer
        
    Returns:
        batch_responses: Batch responses (batch, response_length) or None (if dummy)
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    steps = gen_kwargs["steps"]
    gen_length = gen_kwargs["gen_length"]
    block_length = gen_kwargs["block_length"]
    temperature = gen_kwargs["temperature"]
    remasking = gen_kwargs["remasking"]
    mask_id = gen_kwargs["mask_id"]
    dual_cache = gen_kwargs["dual_cache"]

    batch_start_time = time.time()
    if not dual_cache:
        outputs = generate_with_prefix_cache(
            model=module,
            prompt=idx_repeat,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking, 
            mask_id=mask_id,
            attention_mask=attention_mask_repeat,
        )
    else:
        outputs = generate_with_dual_cache(
            model=module,
            prompt=idx_repeat,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking, 
            mask_id=mask_id,
            attention_mask=attention_mask_repeat,
        )

    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    print(f"[RANK{rank}] FastDLLM generation for {idx_repeat.size(0)} samples cost: {(time.time() - batch_start_time):.2f}s")

    # Process generation outputs
    responses, full_input_ids, attention_mask, answers = process_fastllada_generation_outputs(
        outputs, idx_repeat, attention_mask_repeat, response_length, tokenizer, module.device
    )
    
    return responses, full_input_ids, attention_mask, answers


def process_cjllada_generation_outputs(
    pack: Dict[str, Any],
    outputs: torch.Tensor,
    reversed_traj_unmask_positions: List[torch.Tensor], 
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Process the generation outputs, extract responses, attention_masks and answers
    
    Args:
        pack: Pack info dict
        outputs: Model generation outputs
        reversed_traj_unmask_positions: Unmasked position reversed trajectory
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer
        device: Device
        
    Returns:
        batch_responses: Batch responses (batch, response_length) or None (if dummy)
        batch_reversed_traj: Reversed trajectory of outputs
        batch_reversed_traj_unmask_positions: Unmasked position reversed trajectory
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    if pack["is_dummy"]:
        # directly create placeholder for dummy data
        batch_full_input_ids = torch.full((1, pack["prompt_lengths"][0] + response_length), fill_value=tokenizer.pad_token_id, device=device, dtype=torch.long)
        batch_attention_masks = torch.ones((1, pack["prompt_lengths"][0] + response_length), device=device, dtype=torch.long)
        return None, None, batch_full_input_ids, batch_attention_masks, [["dummy"]]

    # Process real data
    batch_responses = []
    batch_reversed_traj_unmask_positions = []
    batch_answers = []
    batch_attention_masks = []
    cu_seqlens = pack["cu_seqlens"].tolist()
    batch_start_idx = pack["batch_start_idx"]
    prompt_lengths = pack["prompt_lengths"]

    for j in range(pack["num_sequences"]):
        seq_start = cu_seqlens[j]
        valid_prompt_len = prompt_lengths[j]
        response = outputs[0, seq_start + valid_prompt_len:seq_start + valid_prompt_len + response_length].unsqueeze(0)  # (1, response_length)
        reversed_traj_unmask_positions_i = reversed_traj_unmask_positions[0, :, seq_start + valid_prompt_len:seq_start + valid_prompt_len + response_length].unsqueeze(0)  # (1, steps, sequence_length)
        reversed_traj_unmask_positions_i = torch.cat((torch.zeros((1, reversed_traj_unmask_positions_i.size(1), idx_repeat.size(1)), device=device, dtype=reversed_traj_unmask_positions_i.dtype), reversed_traj_unmask_positions_i), dim=2)  # Pad unmask positions for prompt part
        response_mask = torch.ones((1, response_length), device=device)  # response part is all 1
        batch_responses.append(response)
        batch_reversed_traj_unmask_positions.append(reversed_traj_unmask_positions_i)
        batch_attention_masks.append(torch.cat([attention_mask_repeat[batch_start_idx + j].unsqueeze(0), response_mask], dim=1).long())  # (1, sequence_length)
        
        # Extract answers
        response_str = tokenizer.batch_decode(response, skip_special_tokens=True)[0]

        boxed_matches = re.findall(r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}", response_str, re.DOTALL)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
        
        answers = list(set(boxed_matches + answer_matches))  # Merge and deduplicate
        batch_answers.append(answers)

    batch_responses = torch.cat(batch_responses, dim=0)  # (batch, response_length)
    batch_reversed_traj_unmask_positions = torch.cat(batch_reversed_traj_unmask_positions, dim=0)  # (batch, steps, sequence_length)
    batch_full_input_ids = torch.cat([idx_repeat[batch_start_idx:batch_start_idx + pack["num_sequences"]], batch_responses], dim=1)  # (batch, sequence_length)
    batch_attention_masks = torch.cat(batch_attention_masks, dim=0)
    
    return batch_responses, batch_reversed_traj_unmask_positions, batch_full_input_ids, batch_attention_masks, batch_answers


def execute_cjllada_generation(
    pack: Dict[str, Any],
    module: torch.nn.Module,
    gen_kwargs: Dict[str, Any],
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Execute the generation of a single pack, return the generation results
    
    Args:
        pack: Pack info dict
        module: Model
        gen_kwargs: Generation parameters
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer
        
    Returns:
        batch_responses: Batch responses (batch, response_length) or None (if dummy)
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    batch_start_time = time.time()
    outputs, reversed_traj_unmask_positions = reversed_process_llada(
        module,
        pack["packed_input"],
        cu_seqlens=pack["cu_seqlens"],
        max_seqlen=pack["max_seqlen"],
        **gen_kwargs,
    ) # (1, pack_seq_len) (1, steps, pack_seq_len) (1, steps, pack_seq_len)
    print(f"[RANK{dist.get_rank()}] {pack['batch_start_idx']}~{pack['batch_start_idx']+pack['num_sequences']} samples cost: {(time.time() - batch_start_time):.2f}s")

    # Process generation outputs
    batch_responses, batch_reversed_traj_unmask_positions, batch_full_input_ids, batch_attention_masks, batch_answers = process_cjllada_generation_outputs(
        pack, outputs, reversed_traj_unmask_positions, idx_repeat, attention_mask_repeat, response_length, tokenizer, module.device
    )
    
    return batch_responses, batch_reversed_traj_unmask_positions, batch_full_input_ids, batch_attention_masks, batch_answers


def process_fastcjllada_generation_outputs(
    outputs: torch.Tensor,
    reversed_traj_unmask_positions,
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    # Process real data
    batch_size = outputs.size(0)
    prompt_length = idx_repeat.size(1)
    
    # 1. Extract Responses
    # outputs structure is [Prompt, Response]
    # We slice from prompt_length to the end to get responses
    batch_responses = outputs[:, prompt_length:]

    # 2. Full Input IDs
    # The outputs from generate_with_prefix_cache are already the full sequences
    batch_full_input_ids = outputs

    # 3. Attention Masks
    # Concatenate original prompt mask with all-ones mask for response
    # Response part is fully generated, so it's all valid (1)
    response_mask = torch.ones((batch_size, response_length), device=device, dtype=attention_mask_repeat.dtype)
    batch_attention_masks = torch.cat([attention_mask_repeat, response_mask], dim=1)

    # 4. Extract Answers
    batch_answers = []
    
    # Use batch_decode for efficiency
    decoded_responses = tokenizer.batch_decode(batch_responses, skip_special_tokens=True)
    
    for response_str in decoded_responses:
        boxed_matches = re.findall(r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}", response_str, re.DOTALL)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
        
        answers = list(set(boxed_matches + answer_matches))  # Merge and deduplicate
        batch_answers.append(answers)
    
    
    return batch_responses, reversed_traj_unmask_positions, batch_full_input_ids, batch_attention_masks, batch_answers


def execute_fastcjllada_generation(
    module,
    gen_kwargs: Dict[str, Any],
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Execute the generation of a single pack, return the generation results
    
    Args:
        pack: Pack info dict
        module: Model
        gen_kwargs: Generation parameters
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer
        
    Returns:
        batch_responses: Batch responses (batch, response_length) or None (if dummy)
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    steps = gen_kwargs["steps"]
    gen_length = gen_kwargs["gen_length"]
    block_length = gen_kwargs["block_length"]
    temperature = gen_kwargs["temperature"]
    remasking = gen_kwargs["remasking"]
    mask_id = gen_kwargs["mask_id"]
    dual_cache = gen_kwargs["dual_cache"]
    step_merge = gen_kwargs["step_merge"]

    batch_start_time = time.time()
    if not dual_cache:
        outputs, reversed_traj_unmask_positions = fast_reversed_process_llada(
            model=module,
            prompt=idx_repeat,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking, 
            mask_id=mask_id,
            attention_mask=attention_mask_repeat,
            step_merge=step_merge,
        )
    else:
        outputs, reversed_traj_unmask_positions = fast_reversed_process_dual_cache_llada(
            model=module,
            prompt=idx_repeat,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking, 
            mask_id=mask_id,
            attention_mask=attention_mask_repeat,
            step_merge=step_merge,
        )

    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    print(f"[RANK{rank}] FastDLLM generation for {idx_repeat.size(0)} samples cost: {(time.time() - batch_start_time):.2f}s")

    # Process generation outputs
    responses, reversed_traj_unmask_positions, full_input_ids, attention_mask, answers = process_fastcjllada_generation_outputs(
        outputs, reversed_traj_unmask_positions, idx_repeat, attention_mask_repeat, response_length, tokenizer, module.device
    )
    
    return responses, reversed_traj_unmask_positions, full_input_ids, attention_mask, answers


### dream rollout utils ###
def process_dream_generation_outputs(
    outputs: torch.Tensor,
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    # Process real data
    batch_size = outputs.size(0)
    prompt_length = idx_repeat.size(1)
    
    # 1. Extract Responses
    # outputs structure is [Prompt, Response]
    # We slice from prompt_length to the end to get responses
    batch_responses = outputs[:, prompt_length:]

    # 2. Full Input IDs
    # The outputs from generate_with_prefix_cache are already the full sequences
    batch_full_input_ids = outputs

    # 3. Attention Masks
    # Concatenate original prompt mask with all-ones mask for response
    # Response part is fully generated, so it's all valid (1)
    response_mask = torch.ones((batch_size, response_length), device=device, dtype=attention_mask_repeat.dtype)
    batch_attention_masks = torch.cat([attention_mask_repeat, response_mask], dim=1)

    # 4. Extract Answers
    batch_answers = []
    
    # Use batch_decode for efficiency
    decoded_responses = tokenizer.batch_decode(batch_responses, skip_special_tokens=True)
    for response_str in decoded_responses:
        boxed_matches = re.findall(r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}", response_str, re.DOTALL)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
        
        answers = list(set(boxed_matches + answer_matches))  # Merge and deduplicate
        batch_answers.append(answers)
    
    
    return batch_responses, batch_full_input_ids, batch_attention_masks, batch_answers


def execute_dream_generation(
    module: torch.nn.Module,
    gen_kwargs: Dict[str, Any],
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Execute the generation of a single pack, return the generation results
    
    Args:
        pack: Pack info dict
        module: Model
        gen_kwargs: Generation parameters
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer
        
    Returns:
        batch_responses: Batch responses (batch, response_length) or None (if dummy)
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    import types
    from .generation_utils import DreamGenerationMixin
    module.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, module)
    module._sample = types.MethodType(DreamGenerationMixin._sample, module)
    steps = gen_kwargs["steps"]
    gen_length = gen_kwargs["gen_length"]
    block_length = gen_kwargs["block_length"]
    temperature = gen_kwargs["temperature"]
    remasking = gen_kwargs["remasking"]
    mask_id = gen_kwargs["mask_id"]
    do_sample = gen_kwargs["do_sample"]

    batch_start_time = time.time()

    outputs = module.diffusion_generate(
        inputs=idx_repeat,
        attention_mask=attention_mask_repeat,
        max_new_tokens=gen_length,
        output_history=False,
        return_dict_in_generate=False,
        steps=steps,
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        alg="entropy",
        alg_temp=0.0,
        mask_token_id=mask_id,
        do_sample=do_sample,
    )

    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    print(f"[RANK{rank}] DLLM generation for {idx_repeat.size(0)} samples cost: {(time.time() - batch_start_time):.2f}s")

    # Process generation outputs
    responses, full_input_ids, attention_mask, answers = process_dream_generation_outputs(
        outputs, idx_repeat, attention_mask_repeat, response_length, tokenizer, module.device
    )
    
    return responses, full_input_ids, attention_mask, answers


def process_fastdream_generation_outputs(
    outputs: torch.Tensor,
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    # Process real data
    batch_size = outputs.size(0)
    prompt_length = idx_repeat.size(1)
    
    # 1. Extract Responses
    # outputs structure is [Prompt, Response]
    # We slice from prompt_length to the end to get responses
    batch_responses = outputs[:, prompt_length:]

    # 2. Full Input IDs
    # The outputs from generate_with_prefix_cache are already the full sequences
    batch_full_input_ids = outputs

    # 3. Attention Masks
    # Concatenate original prompt mask with all-ones mask for response
    # Response part is fully generated, so it's all valid (1)
    response_mask = torch.ones((batch_size, response_length), device=device, dtype=attention_mask_repeat.dtype)
    batch_attention_masks = torch.cat([attention_mask_repeat, response_mask], dim=1)

    # 4. Extract Answers
    batch_answers = []
    
    # Use batch_decode for efficiency
    decoded_responses = tokenizer.batch_decode(batch_responses, skip_special_tokens=True)
    for response_str in decoded_responses:
        boxed_matches = re.findall(r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}", response_str, re.DOTALL)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
        
        answers = list(set(boxed_matches + answer_matches))  # Merge and deduplicate
        batch_answers.append(answers)
    
    
    return batch_responses, batch_full_input_ids, batch_attention_masks, batch_answers


def execute_fastdream_generation(
    module,
    gen_kwargs: Dict[str, Any],
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Execute the generation of a single pack, return the generation results
    
    Args:
        pack: Pack info dict
        module: Model
        gen_kwargs: Generation parameters
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer
        
    Returns:
        batch_responses: Batch responses (batch, response_length) or None (if dummy)
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    dllm_decode = gen_kwargs.get("dllm_decode", "entropy")
    if dllm_decode == "multiblock":
        from .dream_multiblock import execute_dream_multiblock_generation

        return execute_dream_multiblock_generation(
            module=module,
            gen_kwargs=gen_kwargs,
            idx_repeat=idx_repeat,
            attention_mask_repeat=attention_mask_repeat,
            response_length=response_length,
            tokenizer=tokenizer,
            process_outputs_fn=process_fastdream_generation_outputs,
        )

    import types
    from .generation_utils_block import DreamGenerationMixin
    module.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, module)
    module._sample = types.MethodType(DreamGenerationMixin._sample, module)
    steps = gen_kwargs["steps"]
    gen_length = gen_kwargs["gen_length"]
    block_length = gen_kwargs["block_length"]
    temperature = gen_kwargs["temperature"]
    remasking = gen_kwargs["remasking"]
    mask_id = gen_kwargs["mask_id"]
    do_sample = gen_kwargs["do_sample"]

    batch_start_time = time.time()

    outputs = module.diffusion_generate(
        inputs=idx_repeat,
        attention_mask=attention_mask_repeat,
        max_new_tokens=gen_length,
        output_history=False,
        return_dict_in_generate=False,
        steps=steps,
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        alg="entropy",
        alg_temp=0.0,
        mask_token_id=mask_id,
        do_sample=do_sample,
    )

    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    print(f"[RANK{rank}] DLLM generation for {idx_repeat.size(0)} samples cost: {(time.time() - batch_start_time):.2f}s")

    # Process generation outputs
    responses, full_input_ids, attention_mask, answers = process_fastdream_generation_outputs(
        outputs, idx_repeat, attention_mask_repeat, response_length, tokenizer, module.device
    )
    
    return responses, full_input_ids, attention_mask, answers


def process_cjdream_generation_outputs(
    outputs: torch.Tensor,
    reversed_traj_unmask_positions: torch.Tensor,
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Process Dream CJ generation outputs, extract responses, attention_masks and answers

    Args:
        outputs: Model generation outputs (sequences)
        reversed_traj_unmask_positions: Unmasked position trajectory (may be None for Dream)
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer
        device: Device

    Returns:
        batch_responses: Batch responses (batch, response_length)
        batch_reversed_traj_unmask_positions: Unmasked position trajectory or None
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    batch_size = outputs.size(0)
    prompt_length = idx_repeat.size(1)

    # 1. Extract Responses
    # outputs structure is [Prompt, Response]
    # We slice from prompt_length to end to get responses
    batch_responses = outputs[:, prompt_length:]

    # 2. Full Input IDs
    # The outputs are already full sequences
    batch_full_input_ids = outputs

    # 3. Attention Masks
    # Concatenate original prompt mask with all-ones mask for response
    response_mask = torch.ones((batch_size, response_length), device=device, dtype=attention_mask_repeat.dtype)
    batch_attention_masks = torch.cat([attention_mask_repeat, response_mask], dim=1)

    # 4. Extract Answers
    batch_answers = []

    # Use batch_decode for efficiency
    decoded_responses = tokenizer.batch_decode(batch_responses, skip_special_tokens=True)
    for response_str in decoded_responses:
        boxed_matches = re.findall(r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}", response_str, re.DOTALL)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", response_str, re.DOTALL)

        answers = list(set(boxed_matches + answer_matches))  # Merge and deduplicate
        batch_answers.append(answers)

    batch_reversed_traj_unmask_positions = reversed_traj_unmask_positions

    return batch_responses, batch_reversed_traj_unmask_positions, batch_full_input_ids, batch_attention_masks, batch_answers


def execute_cjdream_generation(
    module: torch.nn.Module,
    gen_kwargs: Dict[str, Any],
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Execute Dream CJ generation, return generation results

    Args:
        module: Model
        gen_kwargs: Generation parameters
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer

    Returns:
        batch_responses: Batch responses (batch, response_length)
        batch_reversed_traj_unmask_positions: Unmasked position trajectory
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    import types
    from .generation_utils import DreamGenerationMixin
    module.cj_diffusion_generate = types.MethodType(DreamGenerationMixin.cj_diffusion_generate, module)
    module._sample = types.MethodType(DreamGenerationMixin._sample, module)

    steps = gen_kwargs.get("steps", 128)
    gen_length = gen_kwargs.get("gen_length", 128)
    block_length = gen_kwargs.get("block_length", 32)
    temperature = gen_kwargs.get("temperature", 0.0)
    remasking = gen_kwargs.get("remasking", "low_confidence")
    mask_id = gen_kwargs.get("mask_id", 151666)
    cfg_scale = gen_kwargs.get("cfg_scale", 0.0)
    mode = gen_kwargs.get("mode", "eval")
    step_merge = gen_kwargs.get("step_merge", False)

    batch_start_time = time.time()

    # Use Dream's reversed process which returns trajectory
    outputs, reversed_traj_unmask_positions = module.cj_diffusion_generate(
        inputs=idx_repeat,
        attention_mask=attention_mask_repeat,
        max_new_tokens=gen_length,
        output_history=False,
        return_dict_in_generate=False,
        steps=steps,
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        alg="entropy",
        alg_temp=0.0,
        mask_token_id=mask_id,
        do_sample=gen_kwargs.get("do_sample", True),
        step_merge=step_merge,
        merge_interval=steps if step_merge else None,
    )

    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    print(f"[RANK{rank}] Dream CJ generation for {idx_repeat.size(0)} samples cost: {(time.time() - batch_start_time):.2f}s")

    # Process generation outputs - now with actual trajectory
    responses, reversed_traj_unmask_positions, full_input_ids, attention_mask, answers = process_cjdream_generation_outputs(
        outputs=outputs,
        reversed_traj_unmask_positions=reversed_traj_unmask_positions,
        idx_repeat=idx_repeat,
        attention_mask_repeat=attention_mask_repeat,
        response_length=response_length,
        tokenizer=tokenizer,
        device=module.device,
    )

    return responses, reversed_traj_unmask_positions, full_input_ids, attention_mask, answers


def process_fast_cjdream_generation_outputs(
    outputs: torch.Tensor,
    reversed_traj_unmask_positions: torch.Tensor,
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Process Dream CJ generation outputs, extract responses, attention_masks and answers

    Args:
        outputs: Model generation outputs (sequences)
        reversed_traj_unmask_positions: Unmasked position trajectory (may be None for Dream)
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer
        device: Device

    Returns:
        batch_responses: Batch responses (batch, response_length)
        batch_reversed_traj_unmask_positions: Unmasked position trajectory or None
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    batch_size = outputs.size(0)
    prompt_length = idx_repeat.size(1)

    # 1. Extract Responses
    # outputs structure is [Prompt, Response]
    # We slice from prompt_length to end to get responses
    batch_responses = outputs[:, prompt_length:]

    # 2. Full Input IDs
    # The outputs are already full sequences
    batch_full_input_ids = outputs

    # 3. Attention Masks
    # Concatenate original prompt mask with all-ones mask for response
    response_mask = torch.ones((batch_size, response_length), device=device, dtype=attention_mask_repeat.dtype)
    batch_attention_masks = torch.cat([attention_mask_repeat, response_mask], dim=1)

    # 4. Extract Answers
    batch_answers = []

    # Use batch_decode for efficiency
    decoded_responses = tokenizer.batch_decode(batch_responses, skip_special_tokens=True)
    for response_str in decoded_responses:
        boxed_matches = re.findall(r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}", response_str, re.DOTALL)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", response_str, re.DOTALL)

        answers = list(set(boxed_matches + answer_matches))  # Merge and deduplicate
        batch_answers.append(answers)

    batch_reversed_traj_unmask_positions = reversed_traj_unmask_positions

    return batch_responses, batch_reversed_traj_unmask_positions, batch_full_input_ids, batch_attention_masks, batch_answers


def execute_fastcjdream_generation(
    module: torch.nn.Module,
    gen_kwargs: Dict[str, Any],
    idx_repeat: torch.Tensor,
    attention_mask_repeat: torch.Tensor,
    response_length: int,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Execute Fast Dream CJ generation, return generation results

    Args:
        module: Model
        gen_kwargs: Generation parameters
        idx_repeat: Repeated input sequences
        attention_mask_repeat: Repeated attention masks
        response_length: Response length
        tokenizer: Tokenizer

    Returns:
        batch_responses: Batch responses (batch, response_length)
        batch_reversed_traj_unmask_positions: Unmasked position trajectory
        batch_full_input_ids: Full input sequences (batch, seq_len)
        batch_attention_masks: Batch attention masks (batch, seq_len)
        batch_answers: Batch answer lists
    """
    import types
    from .generation_utils_block import DreamGenerationMixin
    module.cj_diffusion_generate = types.MethodType(DreamGenerationMixin.cj_diffusion_generate, module)
    module._sample = types.MethodType(DreamGenerationMixin._sample, module)

    steps = gen_kwargs.get("steps", 128)
    gen_length = gen_kwargs.get("gen_length", 128)
    block_length = gen_kwargs.get("block_length", 32)
    temperature = gen_kwargs.get("temperature", 0.0)
    remasking = gen_kwargs.get("remasking", "low_confidence")
    mask_id = gen_kwargs.get("mask_id", 151666)
    step_merge = gen_kwargs.get("step_merge", False)

    batch_start_time = time.time()

    # Use Dream's cj_diffusion_generate method with cache which returns trajectory
    outputs, reversed_traj_unmask_positions = module.cj_diffusion_generate(
        inputs=idx_repeat,
        attention_mask=attention_mask_repeat,
        max_new_tokens=gen_length,
        output_history=False,
        return_dict_in_generate=False,
        steps=steps,
        block_length=block_length,
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        alg="entropy",
        alg_temp=0.0,
        mask_token_id=mask_id,
        do_sample=gen_kwargs.get("do_sample", True),
        step_merge=step_merge,
        merge_interval=steps if step_merge else None,
    )

    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    print(f"[RANK{rank}] Fast Dream CJ generation for {idx_repeat.size(0)} samples cost: {(time.time() - batch_start_time):.2f}s")

    # Process generation outputs - now with actual trajectory
    responses, reversed_traj_unmask_positions, full_input_ids, attention_mask, answers = process_cjdream_generation_outputs(
        outputs=outputs,
        reversed_traj_unmask_positions=reversed_traj_unmask_positions,
        idx_repeat=idx_repeat,
        attention_mask_repeat=attention_mask_repeat,
        response_length=response_length,
        tokenizer=tokenizer,
        device=module.device,
    )

    return responses, reversed_traj_unmask_positions, full_input_ids, attention_mask, answers


### sdar rollout utils ###
def process_sdar_generation_outputs(
    outputs: torch.Tensor,
    idx_repeat: torch.Tensor,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
    # Process real data
    prompt_length = idx_repeat.size(1)
    
    # 1. Extract Responses
    # outputs structure is [Prompt, Response]
    # We slice from prompt_length to the end to get responses
    batch_responses = outputs[:, prompt_length:]

    # 4. Extract Answers
    batch_answers = []
    
    # Use batch_decode for efficiency
    decoded_responses = tokenizer.batch_decode(batch_responses, skip_special_tokens=True)
    for response_str in decoded_responses:
        boxed_matches = re.findall(r"\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}", response_str, re.DOTALL)
        answer_matches = re.findall(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
        
        answers = list(set(boxed_matches + answer_matches))  # Merge and deduplicate
        batch_answers.append(answers)
    
    
    return batch_answers

