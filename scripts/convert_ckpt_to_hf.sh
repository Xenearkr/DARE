#!/bin/bash
#
# Convert FSDP sharded checkpoints to HuggingFace safetensors format.
#
# Usage:
#   bash scripts/convert_ckpt_to_hf.sh \
#       --model_path models/LLaDA-8B-Instruct \
#       --ckpt_path ./ckpts/DARE/<exp>/global_step_40/actor \
#       --output_path ./converted_models/llada_8b_step40 \
#       --model_name llada \
#       --fsdp_strategy fsdp2 \
#       --n_gpus 8
#
# Note: --n_gpus MUST match the number of GPUs used during training
#       (the world_size encoded in checkpoint filenames).
#
set -euo pipefail

# ---------- argument parsing ----------
n_gpus=""
model_path=""
ckpt_path=""
output_path=""
model_name=""
fsdp_strategy="fsdp2"
torch_dtype="bfloat16"
transformer_layer_cls=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_gpus)        n_gpus="$2";              shift 2 ;;
        --model_path)    model_path="$2";           shift 2 ;;
        --ckpt_path)     ckpt_path="$2";            shift 2 ;;
        --output_path)   output_path="$2";          shift 2 ;;
        --model_name)    model_name="$2";           shift 2 ;;
        --fsdp_strategy) fsdp_strategy="$2";        shift 2 ;;
        --torch_dtype)   torch_dtype="$2";          shift 2 ;;
        --transformer_layer_cls) transformer_layer_cls="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------- validate required args ----------
if [[ -z "$model_path" || -z "$ckpt_path" || -z "$output_path" || -z "$model_name" ]]; then
    echo "Error: --model_path, --ckpt_path, --output_path, and --model_name are required."
    echo ""
    echo "Usage:"
    echo "  bash scripts/convert_ckpt_to_hf.sh \\"
    echo "      --model_path models/LLaDA-8B-Instruct \\"
    echo "      --ckpt_path ./ckpts/DARE/<exp>/global_step_40/actor \\"
    echo "      --output_path ./converted_models/llada_8b_step40 \\"
    echo "      --model_name llada \\"
    echo "      --fsdp_strategy fsdp2 \\"
    echo "      --n_gpus 8"
    exit 1
fi

# ---------- auto-detect n_gpus from checkpoint if not specified ----------
if [[ -z "$n_gpus" ]]; then
    # Extract world_size from first model shard filename
    shard_file=$(ls "$ckpt_path"/model_world_size_*_rank_0.pt 2>/dev/null | head -1)
    if [[ -z "$shard_file" ]]; then
        echo "Error: Cannot auto-detect n_gpus. No model shard files found in $ckpt_path"
        echo "Please specify --n_gpus explicitly."
        exit 1
    fi
    n_gpus=$(basename "$shard_file" | sed 's/model_world_size_\([0-9]*\)_rank_.*/\1/')
    echo "[INFO] Auto-detected n_gpus=${n_gpus} from checkpoint filenames"
fi

# ---------- validate checkpoint files exist ----------
for rank in $(seq 0 $((n_gpus - 1))); do
    shard="$ckpt_path/model_world_size_${n_gpus}_rank_${rank}.pt"
    if [[ ! -f "$shard" ]]; then
        echo "Error: Missing shard file: $shard"
        echo "Make sure --n_gpus matches the training world_size and --ckpt_path points to the actor/ directory."
        exit 1
    fi
done
echo "[INFO] All ${n_gpus} shard files found in ${ckpt_path}"

# ---------- build extra args ----------
extra_args=""
if [[ -n "$transformer_layer_cls" ]]; then
    extra_args="--transformer_layer_cls $transformer_layer_cls"
fi

# ---------- launch ----------
echo "[INFO] Launching torchrun with ${n_gpus} processes ..."
echo "[INFO] model_path     = ${model_path}"
echo "[INFO] ckpt_path      = ${ckpt_path}"
echo "[INFO] output_path    = ${output_path}"
echo "[INFO] model_name     = ${model_name}"
echo "[INFO] fsdp_strategy  = ${fsdp_strategy}"
echo "[INFO] torch_dtype    = ${torch_dtype}"
echo ""

torchrun \
    --nproc_per_node="$n_gpus" \
    -m verl.utils.convert_ckpt_to_hf \
    --model_path "$model_path" \
    --ckpt_path "$ckpt_path" \
    --output_path "$output_path" \
    --model_name "$model_name" \
    --fsdp_strategy "$fsdp_strategy" \
    --torch_dtype "$torch_dtype" \
    $extra_args
