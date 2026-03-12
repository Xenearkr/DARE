set -x
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="DARE"
export WANDB_API_KEY=
export WANDB_RESUME="allow"
export WANDB_MODE="offline"
export HF_HOME=
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1

echo "Usage: run_sft_sdar_4b_chat.sh <nproc_per_node> <model_path> [other_configs...]"

nproc_per_node=${1:-8}
MODEL_PATH=${2:-models/SDAR-4B-Chat}

PROJECT_NAME=$WANDB_PROJECT
EXP_NAME="gsm8k-sft-full-sdar-4b-chat"
CKPT_DIR=./ckpts/${PROJECT_NAME}/${EXP_NAME}
LOG_DIR=./logs/${PROJECT_NAME}/${EXP_NAME}
mkdir -p ${CKPT_DIR}
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.sdar_fsdp_sft_trainer \
    data.train_files=data/preprocessed/sft/train/gsm8k_train.parquet \
    data.val_files=data/preprocessed/sft/test/gsm8k_test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=4096 \
    +data.mask_token_id=151669 \
    +data.pad_token_id=151643 \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=${MODEL_PATH} \
    model.trust_remote_code=True \
    +model.attn_implementation="flash_attention_2" \
    +model.fsdp_config.model_dtype=float32 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.logger=["console","wandb"] \
    trainer.total_training_steps=1000 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false
#   \
#   >> ${LOG_DIR}/sft-full-${TIMESTAMP}.out \
#   2>> ${LOG_DIR}/sft-full-${TIMESTAMP}.err &
