#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Add memory fragmentation optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE="offline"
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export SWANLAB_API_KEY=Z2wdWHeNCtCCnPAXsLzX6
export SWANLAB_MODE=cloud

echo "[INFO] Cleaning up old Ray..."
ray stop --force || true
rm -rf /tmp/ray || true

# arguments parsing
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model)
      model="$2"
      shift; shift
      ;;
    --model_path)
      model_path="$2"
      shift; shift
      ;;
    --task)
      task="$2"
      shift; shift
      ;;
    --algorithm)
      algorithm="$2"
      shift; shift
      ;;
    --engine)
      engine="$2"
      shift; shift
      ;;
    *)
      shift
      ;;
  esac
done

algorithm=${algorithm:-dtreerpo}
model=${model:-dream}
model_path=${model_path:-models/Dream-v0-Instruct-7B}
engine=${engine:-hf}

# validate task
valid_tasks=("math" "code" "sudoku" "countdown")
if [[ ! " ${valid_tasks[@]} " =~ " ${task} " ]]; then
    echo "Error: Invalid task '$task'"
    echo "Supported tasks: ${valid_tasks[*]}"
    exit 1
fi

# validate model
valid_models=("llada" "dream")
if [[ ! " ${valid_models[@]} " =~ " ${model} " ]]; then
    echo "Error: Invalid model '$model'"
    echo "Supported models: ${valid_models[*]}"
    exit 1
fi

# Unified hyperparams matching d-TreeRPO reference
max_prompt_length=512
max_response_length=256
num_diffusion_steps=128
total_epoch=10

if [ $task == "math" ]; then
    train_files="['data/preprocessed/rl/train/math_1.parquet','data/preprocessed/rl/train/gsm8k_1.parquet']"
    val_files="['data/preprocessed/rl/test/math500_1.parquet','data/preprocessed/rl/test/gsm8k_1.parquet']"
elif [ $task == "code" ]; then
    train_files="['data/preprocessed/rl/train/lcbv5-K8_1.parquet','data/preprocessed/rl/train/primeintellect-K8_1.parquet','data/preprocessed/rl/train/taco-K8_1.parquet']"
    val_files="['data/preprocessed/rl/test/mbpp_1.parquet','data/preprocessed/rl/test/humaneval_1.parquet','data/preprocessed/rl/test/humanevalplus_1.parquet']"
elif [ $task == "countdown" ]; then
    train_files="['data/preprocessed/rl/train/countdown-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/countdown_1.parquet']"
elif [ $task == "sudoku" ]; then
    train_files="['data/preprocessed/rl/train/sudoku-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/sudoku_1.parquet']"
fi

# Set token IDs based on model
case $model in
    "llada")
        mask_token_id=126336
        pad_token_id=126081
        ;;
    "dream")
        mask_token_id=151666
        pad_token_id=151643
        ;;
    *)
        echo "Error: Unknown model '$model'"
        exit 1
        ;;
esac

# parameters setting
n_gpus_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
batch_size=16  # batch_size must be greater than the number of GPUs used
n_rollout=1    # d-TreeRPO uses tree search instead of standard rollout
lr=3e-5
ppo_micro_batch_size_per_gpu=1
train_temperature=0.9

# d-TreeRPO specific parameters
tree_branch_factor=4          # T: branches per node
tree_contraction_factor=2     # s: spacing factor
num_tree_samples=4            # full trees per prompt
enable_self_distillation=True # self-distillation loss
sd_lambda_max=3e-3            # lambda_max
sd_gamma=2.0                  # gamma
sd_tau_max=2.0                # tau_max
sd_beta=0.7                   # beta

# diffusion related parameters
val_num_diffusion_steps=$max_response_length
block_length=32

timestamp=$(date +"%Y%m%d_%H%M%S")
project_name="DARE"
exp_name="${model}-${task}-dtreerpo-T${tree_branch_factor}-s${tree_contraction_factor}-bsz${batch_size}-prompt${max_prompt_length}-response${max_response_length}-step${num_diffusion_steps}-lr${lr}-temp${train_temperature}-gpu${n_gpus_per_node}-${timestamp}"
ckpt_dir=./ckpts/${project_name}/${exp_name}
log_dir=./logs/${project_name}/${exp_name}
mkdir -p ${ckpt_dir}
mkdir -p ${log_dir}

python3 -m verl.trainer.dllm_main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.name=dtreerpo \
    reward_model.reward_manager=dllm \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.max_resp_len=$max_response_length \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=$batch_size \
    data.val_batch_size=64 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation="error" \
    data.trust_remote_code=True \
    +actor_rollout_ref.algorithm.name=dtreerpo \
    +actor_rollout_ref.model.name=${model} \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.betas="[0.9,0.99]" \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0001 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio=0.4 \
    actor_rollout_ref.actor.grad_clip=0.2 \
    actor_rollout_ref.actor.ppo_epochs=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.lora_rank=128 \
    actor_rollout_ref.model.lora_alpha=64 \
    +actor_rollout_ref.model.lora_dropout=0.05 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.attn_implementation="flash_attention_2" \
    +actor_rollout_ref.model.baseline="${model}-${task}-dtreerpo" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[DreamDecoderLayer] \
    +actor_rollout_ref.actor.mc_num=1 \
    +actor_rollout_ref.actor.n_l=1 \
    +actor_rollout_ref.actor.cfg_scale=0.0 \
    +actor_rollout_ref.actor.baseline="${model}-${task}-dtreerpo" \
    +actor_rollout_ref.actor.tree_branch_factor=$tree_branch_factor \
    +actor_rollout_ref.actor.tree_contraction_factor=$tree_contraction_factor \
    +actor_rollout_ref.actor.num_tree_samples=$num_tree_samples \
    +actor_rollout_ref.actor.enable_self_distillation=$enable_self_distillation \
    +actor_rollout_ref.actor.sd_lambda_max=$sd_lambda_max \
    +actor_rollout_ref.actor.sd_gamma=$sd_gamma \
    +actor_rollout_ref.actor.sd_tau_max=$sd_tau_max \
    +actor_rollout_ref.actor.sd_beta=$sd_beta \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    +actor_rollout_ref.rollout.use_cache=True \
    +actor_rollout_ref.rollout.dual_cache=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.rollout.temperature=$train_temperature \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    +actor_rollout_ref.rollout.val_kwargs.num_diffusion_steps=$val_num_diffusion_steps \
    actor_rollout_ref.rollout.max_num_batched_tokens=11000 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.num_diffusion_steps=$num_diffusion_steps \
    +actor_rollout_ref.rollout.block_length=$block_length \
    +actor_rollout_ref.rollout.mc_num=1 \
    +actor_rollout_ref.rollout.n_l=1 \
    +actor_rollout_ref.rollout.cfg_scale=0.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.default_local_dir=$ckpt_dir \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=$total_epoch \
    custom_reward_function.path="verl/utils/reward_score/__init__.py" \
    custom_reward_function.name="dllm_rm"
