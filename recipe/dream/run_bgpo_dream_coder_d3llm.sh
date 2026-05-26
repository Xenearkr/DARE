#!/bin/bash
# BGPO code RL for d3LLM Dream-Coder via Dream HF rollout + multiblock decode.
# model.name=dream (no d3llm_dream); path points to finetune_d3LLM weights.
set -euo pipefail
set -x

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export WANDB_PROJECT="${WANDB_PROJECT:-DARE}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

smoke_test=0
model_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) model_path="$2"; shift 2 ;;
    --smoke) smoke_test=1; shift ;;
    *) shift ;;
  esac
done

model=dream
algorithm=bgpo
engine=hf
model_path=${model_path:-models/finetune_d3LLM}
task=code
baseline="${model}-${task}-d3llm-${algorithm}-${engine}"

mask_token_id=151666
pad_token_id=151643
block_length=32

if [ "${smoke_test}" -eq 1 ]; then
  train_files="['data/preprocessed/rl/train/lcbv5-K8_1.parquet']"
  val_files="['data/preprocessed/rl/test/humaneval_1.parquet']"
  max_prompt_length=512
  max_response_length=256
  batch_size=4
  n_rollout=2
  mc_num=4
  n_l=4
  ppo_max_token_len_per_gpu=2048
  val_batch_size=8
  save_freq=1
  test_freq=1
  val_before_train=False
  total_epoch=1
  trainer_logger='["console"]'
else
  train_files="['data/preprocessed/rl/train/lcbv5-K8_1.parquet','data/preprocessed/rl/train/primeintellect-K8_1.parquet','data/preprocessed/rl/train/taco-K8_1.parquet']"
  val_files="['data/preprocessed/rl/test/humaneval_1.parquet']"
  max_prompt_length=1024
  max_response_length=512
  batch_size=16
  n_rollout=8
  mc_num=16
  n_l=16
  ppo_max_token_len_per_gpu=5120
  val_batch_size=64
  save_freq=20
  test_freq=20
  val_before_train=False
  total_epoch=5
  trainer_logger='["console","wandb"]'
fi

num_diffusion_steps=${max_response_length}
val_num_diffusion_steps=${max_response_length}
lr=5e-7
ppo_micro_batch_size_per_gpu=1
train_temperature=1.0

n_gpus_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | wc -l)
if [ $((batch_size * n_rollout % n_gpus_per_node)) -ne 0 ]; then
  echo "[ERROR] batch_size * n_rollout must be divisible by GPU count (${n_gpus_per_node})"
  exit 1
fi

echo "[INFO] Ensure Dream modeling files exist (once): bash recipe/d3llm/setup_finetune_d3llm_model_code.sh"

ray stop --force || true
rm -rf /tmp/ray 2>/dev/null || true
ray start --head --node-ip-address=127.0.0.1 --port=6379 \
  --num-gpus="${n_gpus_per_node}" --num-cpus=48

timestamp=$(date +"%Y%m%d_%H%M%S")
smoke_tag=""
[ "${smoke_test}" -eq 1 ] && smoke_tag="-smoke"
exp_name="${baseline}${smoke_tag}-bsz${batch_size}-n${n_rollout}-prompt${max_prompt_length}-response${max_response_length}-bl${block_length}-lr${lr}-temp${train_temperature}-gpu${n_gpus_per_node}-${timestamp}"
ckpt_dir=./ckpts/${WANDB_PROJECT}/${exp_name}
log_dir=./logs/${WANDB_PROJECT}/${exp_name}
mkdir -p "${ckpt_dir}" "${log_dir}"

python3 -m verl.trainer.dllm_main_ppo \
  algorithm.adv_estimator=grpo \
  +algorithm.name=${algorithm} \
  reward_model.reward_manager=dllm \
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
  +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
  data.train_files="${train_files}" \
  data.val_files="${val_files}" \
  data.train_batch_size=${batch_size} \
  data.val_batch_size=${val_batch_size} \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.filter_overlong_prompts=True \
  data.truncation="error" \
  data.trust_remote_code=True \
  +actor_rollout_ref.algorithm.name=${algorithm} \
  +actor_rollout_ref.model.name=${model} \
  actor_rollout_ref.model.path="${model_path}" \
  actor_rollout_ref.actor.optim.lr=${lr} \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${batch_size} \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.model.trust_remote_code=True \
  +actor_rollout_ref.model.attn_implementation="flash_attention_2" \
  +actor_rollout_ref.model.baseline=${baseline} \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[DreamDecoderLayer] \
  +actor_rollout_ref.actor.mc_num=${mc_num} \
  +actor_rollout_ref.actor.n_l=${n_l} \
  +actor_rollout_ref.actor.cfg_scale=0.0 \
  +actor_rollout_ref.actor.baseline=${baseline} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=${engine} \
  +actor_rollout_ref.rollout.use_cache=False \
  +actor_rollout_ref.rollout.dual_cache=False \
  +actor_rollout_ref.rollout.dllm_decode=multiblock \
  +actor_rollout_ref.rollout.d3llm_threshold=0.5 \
  +actor_rollout_ref.rollout.d3llm_block_add_threshold=0.1 \
  +actor_rollout_ref.rollout.d3llm_decoded_token_threshold=0.95 \
  +actor_rollout_ref.rollout.d3llm_cache_delay_iter=32 \
  +actor_rollout_ref.rollout.d3llm_early_stop=True \
  +actor_rollout_ref.rollout.mask_token_id=${mask_token_id} \
  +actor_rollout_ref.rollout.per_sample_seed=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
  actor_rollout_ref.rollout.n=${n_rollout} \
  actor_rollout_ref.rollout.temperature=${train_temperature} \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  +actor_rollout_ref.rollout.val_kwargs.num_diffusion_steps=${val_num_diffusion_steps} \
  actor_rollout_ref.rollout.max_num_batched_tokens=11000 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  +actor_rollout_ref.rollout.num_diffusion_steps=${num_diffusion_steps} \
  +actor_rollout_ref.rollout.block_length=${block_length} \
  +actor_rollout_ref.rollout.mc_num=${mc_num} \
  +actor_rollout_ref.rollout.n_l=${n_l} \
  +actor_rollout_ref.rollout.cfg_scale=0.0 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger="${trainer_logger}" \
  trainer.project_name=${WANDB_PROJECT} \
  trainer.experiment_name=${exp_name} \
  trainer.val_before_train=${val_before_train} \
  trainer.n_gpus_per_node=${n_gpus_per_node} \
  trainer.nnodes=1 \
  trainer.default_local_dir=${ckpt_dir} \
  trainer.save_freq=${save_freq} \
  trainer.test_freq=${test_freq} \
  trainer.total_epochs=${total_epoch} \
  custom_reward_function.path="verl/utils/reward_score/__init__.py" \
  custom_reward_function.name="dllm_rm" \
  2>&1 | tee "${log_dir}/${exp_name}.log"
