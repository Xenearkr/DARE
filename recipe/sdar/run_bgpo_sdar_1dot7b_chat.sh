#!/bin/bash
set -euo pipefail
set -x

# clean residual processes
cleanup() {
    # kill lmdeploy residual processes
    if [ ! -z "${LMDEPLOY_PID+x}" ] && ps -p $LMDEPLOY_PID > /dev/null 2>&1; then
        echo "[INFO] kill lmdeploy process (PID: $LMDEPLOY_PID)..."
        kill -9 $LMDEPLOY_PID 2>/dev/null || true
        sleep 2
    fi

 echo "[INFO] Using engine: ${engine}"
    
    # clean up 23333 port
    local port_pids=$(lsof -ti:23333 2>/dev/null || ss -tlnp | grep :23333 | awk '{print $NF}' | cut -d',' -f2 | sort -u)
    if [ ! -z "$port_pids" ]; then
        for pid in $port_pids; do
            kill -9 $pid 2>/dev/null || true
        done
        sleep 1
    fi

 echo "[INFO] Using engine: ${engine}"
    
    # clean up old ray
    ray stop --force || true
    pkill -f "ray" || true
    rm -rf /tmp/ray || true
}

trap cleanup EXIT INT TERM ERR

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

# clean up old ray
echo "[INFO] Cleaning up old Ray..."
ray stop --force || true
rm -rf /tmp/ray || true

# start up ray head
echo "[INFO] Starting Ray head..."
ray start --head \
  --node-ip-address=127.0.0.1 \
  --port=6379 \
  --num-gpus=8 \
  --num-cpus=128

# start up lmdeploy server (only for lmdeploy engine)
if [ "$engine" = "lmdeploy" ]; then
    echo "[INFO] Starting lmdeploy server on GPU 0..."
    # lmdeploy setting
    MODEL_PATH=${model_path:-models/SDAR-1dot7B-Chat}
    SERVER_PORT=23333
    LMDEPLOY_EXECUTOR_BACKEND="ray"
    LMDEPLOY_ENGINE_BACKEND="pytorch"
    CACHE_MAX_ENTRY_COUNT=0.5
    SESSION_LEN=2048
    BLOCK_LENGTH=4
    UNMASKING_STRATEGY="low_confidence_dynamic"
    DENOISING_STEPS=4
    CONFIDENCE_THRESHOLD=0.9
    export CUDA_VISIBLE_DEVICES=0
    export LMDEPLOY_EXECUTOR_BACKEND=${LMDEPLOY_EXECUTOR_BACKEND}

    lmdeploy serve api_server "${MODEL_PATH}" \
      --backend ${LMDEPLOY_ENGINE_BACKEND} \
      --server-port ${SERVER_PORT} \
      --cache-max-entry-count ${CACHE_MAX_ENTRY_COUNT} \
      --session-len ${SESSION_LEN} \
      --dllm-block-length ${BLOCK_LENGTH} \
      --dllm-unmasking-strategy "${UNMASKING_STRATEGY}" \
      --dllm-denoising-steps ${DENOISING_STEPS} \
      --dllm-confidence-threshold ${CONFIDENCE_THRESHOLD} &

    LMDEPLOY_PID=$!
    echo "[INFO] lmdeploy PID: ${LMDEPLOY_PID}"

    # wait lmdeploy server to be ready
    sleep 5
elif [ "$engine" = "sglang" ]; then
    echo "[INFO] Using SGLang engine (embedded in rollout worker, no separate server needed)"
fi

# env variables for ppo training
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Add memory fragmentation optimization

# Set CUDA_VISIBLE_DEVICES based on engine
if [ "$engine" = "lmdeploy" ]; then
    # GPU 0 for lmdeploy, GPUs 1-7 for training
    export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
elif [ "$engine" = "sglang" ] || [ "$engine" = "hf" ]; then
    # All GPUs available for training
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

export WANDB_PROJECT="DARE"
export WANDB_API_KEY=
export WANDB_RESUME="allow"
export WANDB_MODE="offline"
export HF_HOME=
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

algorithm=${algorithm:-bgpo}
model=${model:-sdar}
model_path=${model_path:-models/SDAR-1dot7B-Chat}
engine=${engine:-lmdeploy}

# validate task
valid_tasks=("math" "code" "sudoku" "countdown")
if [[ ! " ${valid_tasks[@]} " =~ " ${task} " ]]; then
    echo "Error: Invalid task '$task'"
    echo "Supported tasks: ${valid_tasks[*]}"
    exit 1
fi

 echo "[INFO] Using engine: ${engine}"

# validate model
valid_models=("llada" "dream" "sdar")
if [[ ! " ${valid_models[@]} " =~ " ${model} " ]]; then
    echo "Error: Invalid model '$model'"
    echo "Supported models: ${valid_models[*]}"
    exit 1
fi

 echo "[INFO] Using engine: ${engine}"

# validate algorithm
valid_algorithms=("d1" "coupled-grpo" "mdpo" "cj-grpo" "spg" "bgpo")
if [[ ! " ${valid_algorithms[@]} " =~ " ${algorithm} " ]]; then
    echo "Error: Invalid algorithm '$algorithm'"
    echo "Supported algorithms: ${valid_algorithms[*]}"
    exit 1
fi

 echo "[INFO] Using engine: ${engine}"

# validate engine
valid_engines=("hf" "lmdeploy" "sglang")
if [[ ! " ${valid_engines[@]} " =~ " ${engine} " ]]; then
    echo "Error: Invalid engine '$engine'"
    echo "Supported engines: ${valid_engines[*]}"
    exit 1
fi

 echo "[INFO] Using engine: ${engine}"

baseline="${model}-${task}-${algorithm}-${engine}"

if [ $task == "math" ]; then
    train_files="['data/preprocessed/rl/train/math_1.parquet','data/preprocessed/rl/train/gsm8k_1.parquet']"
    val_files="['data/preprocessed/rl/test/math500_1.parquet','data/preprocessed/rl/test/gsm8k_1.parquet']"
    max_prompt_length=512
    max_response_length=512
    total_epoch=1
elif [ $task == "code" ]; then
    train_files="['data/preprocessed/rl/train/lcbv5-K8_1.parquet','data/preprocessed/rl/train/primeintellect-K8_1.parquet','data/preprocessed/rl/train/taco-K8_1.parquet']"
    val_files="['data/preprocessed/rl/test/mbpp_1.parquet','data/preprocessed/rl/test/humaneval_1.parquet','data/preprocessed/rl/test/humanevalplus_1.parquet']"
    max_prompt_length=1024
    max_response_length=512
    total_epoch=5
elif [ $task == "countdown" ]; then
    train_files="['data/preprocessed/rl/train/countdown-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/countdown_1.parquet']"
    max_prompt_length=512
    max_response_length=256
    total_epoch=1
elif [ $task == "sudoku" ]; then
    train_files="['data/preprocessed/rl/train/sudoku-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/sudoku_1.parquet']"
    max_prompt_length=512
    max_response_length=256
    total_epoch=1
fi

 echo "[INFO] Using engine: ${engine}"

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
    "sdar")
        mask_token_id=151669
        pad_token_id=151643
        ;;
    *)
        echo "Error: Unknown model '$model'"
        exit 1
        ;;
esac

# parameters setting
n_gpus_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
batch_size=14  # batch_size must be greater than the number of GPUs used
n_rollout=8
lr=5e-7
ppo_micro_batch_size_per_gpu=1  # gradient accumulation = batch_size / ppo_micro_batch_size_per_gpu
train_temperature=1.0
algorithm="bgpo"

# diffusion related parameters
val_num_diffusion_steps=4
num_diffusion_steps=4
block_length=4
mc_num=16
n_l=16

timestamp=$(date +"%Y%m%d_%H%M%S")
project_name=$WANDB_PROJECT
exp_name="${baseline}-bsz${batch_size}-n${n_rollout}-prompt${max_prompt_length}-response${max_response_length}-step${num_diffusion_steps}-lr${lr}-temp${train_temperature}-n_l${n_l}-mc_num${mc_num}-gpu${n_gpus_per_node}-${timestamp}"
ckpt_dir=./ckpts/${project_name}/${exp_name}
log_dir=./logs/${project_name}/${exp_name}
mkdir -p ${ckpt_dir}
mkdir -p ${log_dir}

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
    data.return_raw_chat=True \
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
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.attn_implementation="flash_attention_2" \
    +actor_rollout_ref.model.baseline=$baseline \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SDARDecoderLayer] \
    +actor_rollout_ref.actor.mc_num=$mc_num \
    +actor_rollout_ref.actor.n_l=$n_l \
    +actor_rollout_ref.actor.cfg_scale=0.0 \
    +actor_rollout_ref.actor.baseline=$baseline \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$engine \
    +actor_rollout_ref.rollout.max_new_tokens=$max_response_length \
    +actor_rollout_ref.rollout.use_cache=False \
    +actor_rollout_ref.rollout.dual_cache=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    +actor_rollout_ref.rollout.mask_token_id=$mask_token_id \
    +actor_rollout_ref.rollout.dllm_algorithm=LowConfidence \
    +actor_rollout_ref.rollout.attention_backend=flashinfer \
    +actor_rollout_ref.rollout.mem_fraction_static=0.6 \
    +actor_rollout_ref.rollout.max_running_requests=1 \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.rollout.temperature=$train_temperature \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.num_diffusion_steps=$val_num_diffusion_steps \
    actor_rollout_ref.rollout.max_num_batched_tokens=11000 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.num_diffusion_steps=$num_diffusion_steps \
    +actor_rollout_ref.rollout.block_length=$block_length \
    +actor_rollout_ref.rollout.mc_num=$mc_num \
    +actor_rollout_ref.rollout.n_l=$n_l \
    +actor_rollout_ref.rollout.cfg_scale=0.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=["console","wandb"] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.val_before_train=False \
    +trainer.val_only=False \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.default_local_dir=$ckpt_dir \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=$total_epoch \
    custom_reward_function.path="verl/utils/reward_score/__init__.py" \
    custom_reward_function.name="dllm_rm" 
    # \
    # 2>&1 | tee ${log_dir}/${baseline}-${timestamp}.log
    
    # >> ${log_dir}/${baseline}-${timestamp}.out \
    # 2>> ${log_dir}/${baseline}-${timestamp}.err &

echo "[INFO] Training finished, stopping lmdeploy..."
kill -9 ${LMDEPLOY_PID}
pkill -f "ray"
# reward_model.reward_manager=dllm: used to select reward_manager in dllm_reward.load_reward_manager()
# llada does not support gradient_checkpointing
# custom_reward_function.name: stored as self.reward_fn, will be called using compute_reward() in ray_trainer

# Enable bfloat16
    # +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    # +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bfloat16 \
    # +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
    # +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \

# Enable LoRA
    # actor_rollout_ref.model.lora_rank=1 \
    # actor_rollout_ref.model.lora_alpha=2 \
    # actor_rollout_ref.model.target_modules=["q_proj","k_proj","v_proj","o_proj","ff_proj","up_proj","down_proj","gate_proj","ff_out"] \
    # +actor_rollout_ref.model.lora_dropout=0.05 \

# Disable dynamic batch size
    # actor_rollout_ref.actor.use_dynamic_bsz=False \
    # actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \

# Enable sequence parallelism, each GPU processes 1/4 of the sequence
    # actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
