set -x
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="DARE"
export WANDB_API_KEY=
export WANDB_RESUME="allow"
export WANDB_MODE="offline"
export HF_HOME=
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1

echo "Usage: run_sft_peft_sdar_30b_a3b_chat.sh <nproc_per_node> <model_path> [other_configs...]"

nproc_per_node=${1:-8}
MODEL_PATH=${2:-models/SDAR-30B-A3B-Chat}

PROJECT_NAME=$WANDB_PROJECT
EXP_NAME="sft-sdar-30b-a3b-chat"
CKPT_DIR=./ckpts/${PROJECT_NAME}/${EXP_NAME}
LOG_DIR=./logs/${PROJECT_NAME}/${EXP_NAME}
mkdir -p ${CKPT_DIR}
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# SDAR-30B-A3B-Chat (MoE) specific configuration:
# - 128 experts, 8 experts per token
# - Layer-wise expert parallel storage (layer-*-ep-0-of-1.safetensors)
# - Uses size-based FSDP wrapping for MoE (no expert sharding)
# - mask_token_id: 151643 (from SDAR config, same as eos_token_id)

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.sdar_moe_fsdp_sft_trainer \
    data.train_files=data/preprocessed/sft/train/gsm8k_train.parquet \
    data.val_files=data/preprocessed/sft/test/gsm8k_test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=4096 \
    +data.mask_token_id=151643 \
    +data.pad_token_id=151643 \
    optim.lr=5e-5 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=8 \
    model.partial_pretrain=${MODEL_PATH} \
    model.trust_remote_code=True \
    +model.block_size=4 \
    +model.attn_implementation="flash_attention_2" \
    +model.fsdp_config.model_dtype=bfloat16 \
    model.fsdp_config.cpu_offload=false \
    model.fsdp_config.offload_params=false \
    model.fsdp_config.wrap_policy.min_num_params=10000 \
    model.enable_gradient_checkpointing=true \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.logger=["console","wandb"] \
    trainer.total_epochs=1 \
    trainer.total_training_steps=500 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    model.lora_rank=16 \
    model.lora_alpha=8 \
    model.target_modules=all-linear \
    >> ${LOG_DIR}/sft-${TIMESTAMP}.out \
    2>> ${LOG_DIR}/sft-${TIMESTAMP}.err &

# Note: For 30B MoE model, we use smaller batch size and learning rate
# to fit within GPU memory and ensure stable training.
