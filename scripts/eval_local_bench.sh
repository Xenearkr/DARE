#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Add memory fragmentation optimization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="DARE_EVAL_CACHE"
export WANDB_API_KEY=42598cc56636f040038970a197ecd2c231a697cc
export WANDB_RESUME="allow"
export WANDB_MODE="offline"
export HF_HOME=/mnt/shared-storage-user/yangjingyi/huggingface
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

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
    --ckpt_dir)
      ckpt_dir="$2"
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

algorithm=${algorithm:-bgpo}
model=${model:-llada}
model_path=${model_path:-/mnt/shared-storage-user/yangjingyi/BGPO/models/LLaDA-8B-Instruct}
engine=${engine:-hf}

# validate task
valid_tasks=("math" "code" "sudoku" "countdown")
if [[ ! " ${valid_tasks[@]} " =~ " ${task} " ]]; then
    echo "Error: Invalid task '$task'"
    echo "Supported tasks: ${valid_tasks[*]}"
    exit 1
fi

# validate model
valid_models=("llada" "dream" "sdar")
if [[ ! " ${valid_models[@]} " =~ " ${model} " ]]; then
    echo "Error: Invalid model '$model'"
    echo "Supported models: ${valid_models[*]}"
    exit 1
fi

# validate algorithm
valid_algorithms=("d1" "coupled-grpo" "mdpo" "cj-grpo" "spg" "bgpo")
if [[ ! " ${valid_algorithms[@]} " =~ " ${algorithm} " ]]; then
    echo "Error: Invalid algorithm '$algorithm'"
    echo "Supported algorithms: ${valid_algorithms[*]}"
    exit 1
fi

# validate engine
valid_engines=("hf" "lmdeploy")
if [[ ! " ${valid_engines[@]} " =~ " ${engine} " ]]; then
    echo "Error: Invalid engine '$engine'"
    echo "Supported engines: ${valid_engines[*]}"
    exit 1
fi

baseline="${model}-${task}-${algorithm}-${engine}"

if [ $task == "math" ]; then
    train_files="['data/preprocessed/rl/train/math_1.parquet','data/preprocessed/rl/train/gsm8k_1.parquet']"
    val_files="['data/preprocessed/rl/test/math500_1.parquet','data/preprocessed/rl/test/gsm8k_1.parquet']"
    max_prompt_length=512
    max_response_length=512
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
elif [ $task == "code" ]; then
    train_files="['data/preprocessed/rl/train/lcbv5-K8_1.parquet','data/preprocessed/rl/train/primeintellect-K8_1.parquet','data/preprocessed/rl/train/taco-K8_1.parquet']"
    val_files="['data/preprocessed/rl/test/mbpp_1.parquet','data/preprocessed/rl/test/humaneval_1.parquet','data/preprocessed/rl/test/humanevalplus_1.parquet']"
    max_prompt_length=1024
    max_response_length=512
    num_diffusion_steps=$max_response_length
    total_epoch=5
elif [ $task == "countdown" ]; then
    train_files="['data/preprocessed/rl/train/countdown-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/countdown_1.parquet']"
    max_prompt_length=512
    max_response_length=256
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
elif [ $task == "sudoku" ]; then
    train_files="['data/preprocessed/rl/train/sudoku-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/sudoku_1.parquet']"
    max_prompt_length=512
    max_response_length=256
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
fi

# Set token IDs and transformer_layer_cls_to_wrap based on model
case $model in
    "llada")
        mask_token_id=126336
        pad_token_id=126081
        transformer_layer_cls_to_wrap="LLaDALlamaBlock"
        ;;
    "dream")
        mask_token_id=151666
        pad_token_id=151643
        transformer_layer_cls_to_wrap="DreamDecoderLayer"
        ;;
    "sdar")
        mask_token_id=151669
        pad_token_id=151643
        transformer_layer_cls_to_wrap="SDARDecoderLayer"
        ;;
    *)
        echo "Error: Unknown model '$model'"
        exit 1
        ;;
esac

# parameters setting
n_gpus_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
batch_size=8  # batch_size must be greater than the number of GPUs used
n_rollout=8
lr=5e-7
ppo_micro_batch_size_per_gpu=1  # gradient accumulation = batch_size / ppo_micro_batch_size_per_gpu

# Temperature settings: use 0.2 for dLLMs (Dream/LLaDA) to match paper results
# Temperature 0.0 (greedy decoding) significantly reduces quality for diffusion models
# Reference: Dream recipe scripts use temperature=0.2 for both training and validation
if [ "$model" == "dream" ]; then
    train_temperature=0.6
    val_temperature=0.2
else
    train_temperature=0.6
    val_temperature=0.0
fi

# diffusion related parameters
val_num_diffusion_steps=$max_response_length
block_length=32
mc_num=1
n_l=1
use_cache=True
step_merge=False

timestamp=$(date +"%Y%m%d_%H%M%S")
project_name=$WANDB_PROJECT
exp_name=$(echo "$model_path" | awk -F'/' '{print $(NF-1)"/"$NF}')
ckpt_dir=$ckpt_dir
log_dir=./logs/${project_name}/${exp_name}
mkdir -p ${log_dir}

# Build resume args based on whether ckpt_dir is specified
if [ -n "$ckpt_dir" ]; then
    resume_args="trainer.resume_mode=resume_path trainer.resume_from_path=$ckpt_dir trainer.default_local_dir=$ckpt_dir"
    mkdir -p ${ckpt_dir}
    log_suffix=$(echo "$ckpt_dir" | awk -F'/' '{print $(NF-1)"-"$NF}')
else
    resume_args="trainer.default_local_dir=$ckpt_dir"
    log_suffix=$(echo "$model_path" | awk -F'/' '{print $(NF-1)"-"$NF}')
fi

# SPG algorithm specific parameters
if [ "$algorithm" == "spg" ]; then
    logp_estimation="mix"
    actor_config_args="+actor_rollout_ref.actor.logp_estimation=$logp_estimation"
else
    actor_config_args=""
fi

python3 -m verl.trainer.dllm_main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.name=${algorithm} \
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
    +actor_rollout_ref.algorithm.name=${algorithm} \
    +actor_rollout_ref.model.name=${model} \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    ${actor_config_args} \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.attn_implementation="flash_attention_2" \
    +actor_rollout_ref.model.baseline=$baseline \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[$transformer_layer_cls_to_wrap] \
    +actor_rollout_ref.actor.mc_num=$mc_num \
    +actor_rollout_ref.actor.n_l=$n_l \
    +actor_rollout_ref.actor.cfg_scale=0.0 \
    +actor_rollout_ref.actor.baseline=$baseline \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    +actor_rollout_ref.rollout.use_cache=$use_cache \
    +actor_rollout_ref.rollout.dual_cache=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.rollout.temperature=$train_temperature \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    +actor_rollout_ref.rollout.val_kwargs.num_diffusion_steps=$val_num_diffusion_steps \
    actor_rollout_ref.rollout.max_num_batched_tokens=11000 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.num_diffusion_steps=$num_diffusion_steps \
    +actor_rollout_ref.rollout.block_length=$block_length \
    +actor_rollout_ref.rollout.mc_num=$mc_num \
    +actor_rollout_ref.rollout.n_l=$n_l \
    +actor_rollout_ref.rollout.cfg_scale=0.0 \
    +actor_rollout_ref.rollout.step_merge=${step_merge} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=["console","wandb"] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=True \
    +trainer.val_only=True \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    $resume_args \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=$total_epoch \
    custom_reward_function.path="verl/utils/reward_score/__init__.py" \
    custom_reward_function.name="dllm_rm" \
    >> ${log_dir}/${log_suffix}.out \
    2>> ${log_dir}/${log_suffix}.err

