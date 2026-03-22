set -x
export HYDRA_FULL_ERROR=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Add memory fragmentation optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="DARE"
export WANDB_API_KEY=
export WANDB_RESUME="allow"
export WANDB_MODE="offline"
export HF_HOME=
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export LLADA2_SFT_DEBUG=0
export LLADA2_SFT_DEBUG_MAX_STEPS=64

echo "Usage: run_sft_peft_llada2_mini.sh <nproc_per_node> <model_path> [other_configs...]"

nproc_per_node=${1:-8}
MODEL_PATH=${2:-models/LLaDA2.0-mini}

PROJECT_NAME=$WANDB_PROJECT
EXP_NAME="sft-llada2-mini"
CKPT_DIR=./ckpts/${PROJECT_NAME}/${EXP_NAME}
LOG_DIR=./logs/${PROJECT_NAME}/${EXP_NAME}
mkdir -p ${CKPT_DIR}
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# LLaDA2.0-mini (MoE) specific configuration:
# - 256 experts, 8 experts per token, 1 shared expert
# - mask_token_id: 156895 (from config)
# - pad_token_id: 156892 (from config)
# - Uses size-based FSDP wrapping for MoE (no expert sharding)

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.llada2_fsdp_sft_trainer \
    data.train_files=data/preprocessed/sft/train/gsm8k_train.parquet \
    data.val_files=data/preprocessed/sft/test/gsm8k_test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=4096 \
    +data.mask_token_id=156895 \
    +data.pad_token_id=156892 \
    +data.noise_range_low=0.3 \
    +data.noise_range_high=0.8 \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=${MODEL_PATH} \
    model.trust_remote_code=True \
    +model.attn_implementation="sdpa" \
    +model.fsdp_config.model_dtype=float16 \
    model.fsdp_config.wrap_policy.min_num_params=10000 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.logger=["console","wandb"] \
    trainer.total_training_steps=1000 \
    +trainer.block_diffusion_mode=true \
    +trainer.block_size=32 \
    +trainer.same_token_labels=true \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    >> ${LOG_DIR}/sft-${TIMESTAMP}.out \
    2>> ${LOG_DIR}/sft-${TIMESTAMP}.err &

# Note: For MoE models, we use min_num_params instead of transformer_layer_cls_to_wrap
# to avoid sharding expert weights, which would incur significant communication overhead.
# LLaDA2.0-mini local implementation does not support flash_attention_2.
