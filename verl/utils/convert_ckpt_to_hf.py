"""
Convert FSDP sharded checkpoints to HuggingFace safetensors format.

Launch with torchrun (n_gpus must match the training world_size):
    torchrun --nproc_per_node=8 scripts/convert_ckpt_to_hf.py \
        --model_path models/LLaDA-8B-Instruct \
        --ckpt_path ./ckpts/DARE/<exp>/global_step_40/actor \
        --output_path ./converted_models/llada_8b_step40 \
        --model_name llada \
        --fsdp_strategy fsdp2
"""

import argparse
import os
import warnings

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardedStateDictConfig,
    StateDictType,
)

from verl.utils.fsdp_utils import (
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_state_ctx,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)


# ---------- default layer class names per model ----------
DEFAULT_LAYER_CLS = {
    "llada": ["LLaDALlamaBlock"],
    "dream": ["DreamDecoderLayer"],
    "sdar": ["SDARDecoderLayer"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert FSDP sharded ckpt → HuggingFace safetensors")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the original pretrained model (for config/tokenizer/architecture)")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the FSDP sharded checkpoint directory (the actor/ folder)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for the converted HuggingFace model")
    parser.add_argument("--model_name", type=str, required=True, choices=["llada", "dream", "sdar"],
                        help="Model type (determines AutoModel class and default layer cls)")
    parser.add_argument("--fsdp_strategy", type=str, default="fsdp2", choices=["fsdp", "fsdp2"],
                        help="FSDP strategy used during training (default: fsdp2)")
    parser.add_argument("--transformer_layer_cls", type=str, nargs="*", default=None,
                        help="Override transformer layer class names for FSDP wrap policy "
                             "(auto-detected from model_name by default)")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16", "float16"],
                        help="Dtype for saving the model (default: bfloat16)")
    return parser.parse_args()


def detect_world_size(ckpt_path: str) -> int:
    """Auto-detect the training world_size from checkpoint filenames."""
    for f in os.listdir(ckpt_path):
        if f.startswith("model_world_size_") and f.endswith(".pt"):
            # e.g. model_world_size_8_rank_0.pt
            parts = f.replace(".pt", "").split("_")
            ws_idx = parts.index("size") + 1
            return int(parts[ws_idx])
    raise FileNotFoundError(f"No model shard files found in {ckpt_path}")


def build_model(model_path: str, model_name: str, torch_dtype: torch.dtype, device_mesh):
    """Build the HF model (same as training) without optimizer."""
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    init_context = get_init_weight_context_manager(
        use_meta_tensor=not actor_model_config.tie_word_embeddings,
        mesh=device_mesh,
    )

    if model_name in ("llada", "dream"):
        auto_cls = AutoModel
    else:  # sdar
        auto_cls = AutoModelForCausalLM

    with init_context(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = auto_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            config=actor_model_config,
            trust_remote_code=True,
        )
        model.to(torch_dtype)

    return model, actor_model_config


def apply_fsdp_wrapping(model, fsdp_strategy: str, layer_cls_names: list, device_mesh, torch_dtype: torch.dtype):
    """Apply FSDP wrapping to the model (same as training)."""
    wrap_policy_config = {"transformer_layer_cls_to_wrap": layer_cls_names}
    auto_wrap_policy = get_fsdp_wrap_policy(module=model, config=wrap_policy_config)

    if fsdp_strategy == "fsdp":
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        model = FSDP(
            model,
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=device_mesh,
        )
    elif fsdp_strategy == "fsdp2":
        from packaging import version
        if version.parse(torch.__version__) >= version.parse("2.6"):
            from torch.distributed.fsdp import MixedPrecisionPolicy
        else:
            from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        fsdp_kwargs = {
            "mesh": device_mesh,
            "mp_policy": mp_policy,
            "reshard_after_forward": True,
        }
        full_state = model.state_dict()
        fsdp_config = {"wrap_policy": {"transformer_layer_cls_to_wrap": layer_cls_names}}
        apply_fsdp2(model, fsdp_kwargs, fsdp_config)
        fsdp2_load_full_state_dict(model, full_state, device_mesh)

    return model


def load_sharded_checkpoint(model, ckpt_path: str):
    """Load per-rank sharded model checkpoint into the FSDP model."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_file = os.path.join(ckpt_path, f"model_world_size_{world_size}_rank_{rank}.pt")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"[rank-{rank}] Shard file not found: {model_file}")

    print(f"[rank-{rank}] Loading shard from {model_file}")
    model_state_dict = torch.load(model_file, map_location="cpu", weights_only=False)

    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with get_fsdp_state_ctx(model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, None):
        model.load_state_dict(model_state_dict)

    del model_state_dict
    torch.cuda.empty_cache()
    dist.barrier()
    print(f"[rank-{rank}] Shard loaded successfully")


def consolidate_and_save(model, model_config, model_path: str, output_path: str, torch_dtype: torch.dtype):
    """Consolidate full state dict on rank 0 and save as HuggingFace format."""
    rank = dist.get_rank()

    print(f"[rank-{rank}] Consolidating full state dict ...")

    if fsdp_version(model) == 1:
        # FSDP1: use FullStateDictConfig
        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg, None):
            full_state = model.state_dict()
    else:
        # FSDP2: use DCP get_model_state_dict
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        full_state = get_model_state_dict(model, options=options)

    if rank == 0:
        print(f"[rank-0] Full state dict consolidated, {len(full_state)} keys")
        print(f"[rank-0] Saving HuggingFace model to {output_path} ...")

        os.makedirs(output_path, exist_ok=True)

        from accelerate import init_empty_weights
        from transformers import AutoModel, AutoModelForCausalLM

        # Determine the auto model class from architecture
        arch = model_config.architectures[0] if hasattr(model_config, "architectures") and model_config.architectures else ""
        if "ForCausalLM" in arch:
            auto_cls = AutoModelForCausalLM
        else:
            auto_cls = AutoModel

        with init_empty_weights():
            save_model = auto_cls.from_config(model_config, torch_dtype=torch_dtype)
        save_model.to_empty(device="cpu")

        # save model weights as safetensors
        save_model.save_pretrained(output_path, state_dict=full_state)

        # save tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)

        # copy custom modeling files if they exist (needed for trust_remote_code models)
        _copy_custom_modeling_files(model_path, output_path)

        del full_state
        del save_model
        print(f"[rank-0] Done! Model saved to {output_path}")
    else:
        del full_state

    dist.barrier()


def _copy_custom_modeling_files(src_dir: str, dst_dir: str):
    """Copy custom modeling_*.py files needed for trust_remote_code models."""
    import shutil
    for f in os.listdir(src_dir):
        if f.startswith("modeling_") and f.endswith(".py"):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            print(f"[rank-0] Copied {f} to output directory")
        elif f == "configuration_llada.py" or (f.startswith("configuration_") and f.endswith(".py")):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            print(f"[rank-0] Copied {f} to output directory")


def main():
    args = parse_args()

    # --- 0. Validate checkpoint world_size matches current launch ---
    ckpt_world_size = detect_world_size(args.ckpt_path)

    # --- 1. Initialize distributed ---
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    if world_size != ckpt_world_size:
        if rank == 0:
            print(f"ERROR: Current world_size ({world_size}) != checkpoint world_size ({ckpt_world_size}).")
            print(f"Please launch with: torchrun --nproc_per_node={ckpt_world_size} ...")
        dist.destroy_process_group()
        return

    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

    if rank == 0:
        print(f"=== FSDP Checkpoint -> HuggingFace Converter ===")
        print(f"  model_path     : {args.model_path}")
        print(f"  ckpt_path      : {args.ckpt_path}")
        print(f"  output_path    : {args.output_path}")
        print(f"  model_name     : {args.model_name}")
        print(f"  fsdp_strategy  : {args.fsdp_strategy}")
        print(f"  world_size     : {world_size}")
        print(f"  torch_dtype    : {args.torch_dtype}")

    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.torch_dtype]

    layer_cls_names = args.transformer_layer_cls or DEFAULT_LAYER_CLS[args.model_name]
    if rank == 0:
        print(f"  layer_cls      : {layer_cls_names}")

    # --- 2. Build model with FSDP ---
    if rank == 0:
        print("\n[Step 1/3] Building model with FSDP wrapping ...")
    model, model_config = build_model(args.model_path, args.model_name, torch_dtype, device_mesh)
    dist.barrier()
    model = apply_fsdp_wrapping(model, args.fsdp_strategy, layer_cls_names, device_mesh, torch_dtype)
    dist.barrier()
    if rank == 0:
        print("[Step 1/3] Model built and FSDP wrapped")

    # --- 3. Load sharded checkpoint ---
    if rank == 0:
        print("\n[Step 2/3] Loading sharded checkpoint ...")
    load_sharded_checkpoint(model, args.ckpt_path)
    if rank == 0:
        print("[Step 2/3] Checkpoint loaded")

    # --- 4. Consolidate and save ---
    if rank == 0:
        print("\n[Step 3/3] Consolidating and saving HuggingFace model ...")
    consolidate_and_save(model, model_config, args.model_path, args.output_path, torch_dtype)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
