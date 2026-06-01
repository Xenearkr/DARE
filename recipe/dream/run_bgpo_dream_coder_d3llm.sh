#!/bin/bash
# BGPO code RL for d3LLM Dream-Coder (HF or SGLang multiblock rollout).
# model.name=dream (no d3llm_dream); path points to finetune_d3LLM weights.
#
# Usage:
#   bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --smoke              # default engine: sglang
#   bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --smoke --engine hf
#   bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --engine sglang
set -euo pipefail
set -x

cleanup() {
    ray stop --force || true
    pkill -u "$(whoami)" -f "dllm_main_ppo" 2>/dev/null || true
    rm -rf /tmp/ray 2>/dev/null || true
}
trap cleanup EXIT INT TERM ERR

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export WANDB_PROJECT="${WANDB_PROJECT:-DARE}"
export WANDB_API_KEY="${WANDB_API_KEY:-wandb_v1_ZyTW8NCbOruLfue0ZyzHc7XoUoz}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE=1

# Suppress known noisy third-party warnings in long training logs.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
DARE_SUPPRESS_WARNINGS="ignore:pkg_resources is deprecated:UserWarning,ignore:The pynvml package is deprecated:FutureWarning"
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}${DARE_SUPPRESS_WARNINGS}"

# Prefer DARE conda env; ignore an active base/other conda (CONDA_PREFIX mismatch breaks SGLang JIT).
DARE_ENV="${HOME}/anaconda3/envs/DARE"
if [[ -x "${DARE_ENV}/bin/python" ]]; then
  PYTHON="${PYTHON:-${DARE_ENV}/bin/python}"
  CONDA_PREFIX="${DARE_ENV}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
  CONDA_PREFIX="${CONDA_PREFIX:-}"
fi
export CONDA_PREFIX
export PATH="$(dirname "${PYTHON}"):${PATH}"

smoke_test=0
model_path=""
engine=sglang

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) model_path="$2"; shift 2 ;;
    --engine) engine="$2"; shift 2 ;;
    --smoke) smoke_test=1; shift ;;
    *) echo "[WARN] Unknown arg: $1"; shift ;;
  esac
done

valid_engines=(hf sglang)
if [[ ! " ${valid_engines[*]} " =~ " ${engine} " ]]; then
  echo "[ERROR] Invalid engine '${engine}'. Supported: ${valid_engines[*]}"
  exit 1
fi

model=dream
algorithm=bgpo
model_path=${model_path:-models/finetune_d3LLM}
task=code
baseline="${model}-${task}-d3llm-${algorithm}-${engine}"

mask_token_id=151666
pad_token_id=151643
block_length=32

# SGLang memory_saver conflicts with expandable_segments; set before ray start.
sglang_mem_fraction_static=0.45
sglang_gpu_memory_utilization=0.4
sglang_attention_backend=fa3
sglang_disable_cuda_graph=False
actor_param_offload=False
actor_optimizer_offload=False
enable_activation_offload=False

ORIG_TRAIN_FILES="['data/preprocessed/rl/train/lcbv5-K8_1.parquet','data/preprocessed/rl/train/primeintellect-K8_1.parquet','data/preprocessed/rl/train/taco-K8_1.parquet']"

if [ "${smoke_test}" -eq 1 ]; then
  # Override stale single-GPU env left by benchmark scripts (default smoke uses 4 GPUs).
  export CUDA_VISIBLE_DEVICES="${DARE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  train_files="${ORIG_TRAIN_FILES}"
  EVALPLUS_SMOKE_VAL_PARQUET="data/preprocessed/rl/test/humaneval_evalplus_smoke_8.parquet"
  if [ ! -f "${EVALPLUS_SMOKE_VAL_PARQUET}" ]; then
    echo "[INFO] Building smoke val subset (8 samples): ${EVALPLUS_SMOKE_VAL_PARQUET}"
    "${PYTHON}" - <<'PY'
import pandas as pd
src = "data/preprocessed/rl/test/humaneval_evalplus_1.parquet"
dst = "data/preprocessed/rl/test/humaneval_evalplus_smoke_8.parquet"
pd.read_parquet(src).head(8).to_parquet(dst, index=False)
print(f"wrote {dst}")
PY
  fi
  val_files="['${EVALPLUS_SMOKE_VAL_PARQUET}']"
  max_prompt_length=1024
  # Align with d3LLM evalplus: max_new_tokens=512, temperature=0.0 (greedy).
  max_response_length=512
  batch_size=4
  n_rollout=4
  mc_num=4
  n_l=4
  ppo_max_token_len_per_gpu=2048
  max_num_batched_tokens=4096
  val_batch_size=32
  save_freq=0
  # HumanEval before/after 1 train step: val_generations/0.jsonl (base) vs 1.jsonl (step1).
  test_freq=1
  val_before_train=True
  total_epoch=1
  trainer_logger='["console","wandb"]'
  enable_gradient_checkpointing=False
  smoke_total_training_steps=1
  train_temperature=0.4
  val_temperature=0.0
  val_do_sample=False
  if [ "$engine" = "sglang" ]; then
    # Leave headroom on 48GB for FSDP state_dict + weight sync beside SGLang static pool.
    sglang_mem_fraction_static=0.32
    sglang_gpu_memory_utilization=0.32
    sglang_attention_backend=torch_native
    sglang_disable_cuda_graph=True
    actor_param_offload=True
    actor_optimizer_offload=True
    enable_activation_offload=True
  fi
else
  # Full training (aligned with recipe/sdar/run_bgpo_sdar_8b_chat.sh sglang branch).
  train_files="${ORIG_TRAIN_FILES}"
  val_files="['data/preprocessed/rl/test/humaneval_evalplus_1.parquet']"
  max_prompt_length=1024
  # Dream d3LLM multiblock: keep 512 (SDAR code full uses 1536; too costly per-sample here).
  max_response_length=768
  batch_size=8
  n_rollout=4
  mc_num=8
  n_l=8
  ppo_max_token_len_per_gpu=3072
  max_num_batched_tokens=6144
  val_batch_size=32
  save_freq=40
  test_freq=20
  val_before_train=True
  total_epoch=1
  trainer_logger='["console","wandb"]'
  enable_gradient_checkpointing=True
  if [ "$engine" = "sglang" ]; then
    echo "[INFO] SGLang full run: using smoke-style inference mem + activation offload"
    sglang_mem_fraction_static=0.32
    sglang_gpu_memory_utilization=0.32
    sglang_attention_backend=torch_native
    sglang_disable_cuda_graph=True
    actor_param_offload=True
    actor_optimizer_offload=True
    enable_activation_offload=True
  fi
fi

if [ "$engine" = "sglang" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
  _cuda_rt_lib="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
  _cuda_targets_lib="${CONDA_PREFIX}/targets/x86_64-linux/lib"
  export CUDA_HOME="${CONDA_PREFIX}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${_cuda_rt_lib}:${_cuda_targets_lib}:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${CONDA_PREFIX}/lib:${_cuda_rt_lib}:${_cuda_targets_lib}:${LIBRARY_PATH:-}"
  if [[ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" ]]; then
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
    export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
  fi
  echo "[INFO] SGLang build env: CONDA_PREFIX=${CONDA_PREFIX} CXX=${CXX:-unset} LIBRARY_PATH=${LIBRARY_PATH}"
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

num_diffusion_steps=${max_response_length}
val_num_diffusion_steps=${max_response_length}
lr=5e-7
ppo_micro_batch_size_per_gpu=1
# Train rollout temperature (smoke block may override).
train_temperature="${train_temperature:-0.4}"
# Keep validation slightly cooler for stable HumanEval monitoring.
val_temperature="${val_temperature:-0.0}"
val_top_p="${val_top_p:-1.0}"
val_do_sample="${val_do_sample:-False}"

# Certainty-Forcing Loss (d3LLM distill alignment)
enable_cfl=True
cfl_coef=0.5
cfl_temperature=0.5
cfl_gate_positive_adv_only=False
if [ "${smoke_test}" -eq 1 ]; then
  # Smoke: gate A — all rollout samples, verify plumbing and metrics.
  cfl_gate_passed_only=False
else
  # Full: gate B — only passed rollouts (requires token_level_scores).
  cfl_gate_passed_only=True
fi

n_gpus_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | wc -l)
real_train_batch_size=$((batch_size * n_rollout))
if [ $((real_train_batch_size % n_gpus_per_node)) -ne 0 ]; then
  echo "[ERROR] batch_size(${batch_size}) * n_rollout(${n_rollout}) = ${real_train_batch_size} must be divisible by GPU count (${n_gpus_per_node})"
  exit 1
fi
if [ $((mc_num % n_l)) -ne 0 ]; then
  echo "[ERROR] mc_num(${mc_num}) must be divisible by n_l(${n_l})"
  exit 1
fi

echo "[INFO] engine=${engine} smoke=${smoke_test} GPUs=${n_gpus_per_node} train_temperature=${train_temperature}"
echo "[INFO] CFL: enable=${enable_cfl} coef=${cfl_coef} temperature=${cfl_temperature} gate_passed=${cfl_gate_passed_only}"
echo "[INFO] HumanEval eval: max_response=${max_response_length} temperature=${val_temperature} top_p=${val_top_p} do_sample=${val_do_sample}"
echo "[INFO] W&B val metric: val-core/humaneval/acc/mean@1; val_before_train=${val_before_train}"
echo "[INFO] Ensure Dream modeling files exist (once): bash recipe/d3llm/setup_finetune_d3llm_model_code.sh"
EVALPLUS_VAL_PARQUET="data/preprocessed/rl/test/humaneval_evalplus_1.parquet"
if [[ ! -f "${EVALPLUS_VAL_PARQUET}" ]]; then
  echo "[INFO] Building EvalPlus val parquet: ${PYTHON} recipe/d3llm/build_evalplus_code_mix.py"
  "${PYTHON}" recipe/d3llm/build_evalplus_code_mix.py || {
    echo "[ERROR] Failed to build EvalPlus val parquet."
    exit 1
  }
fi
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
export DREAM_ROLLOUT_VERBOSE="${DREAM_ROLLOUT_VERBOSE:-1}"
export DREAM_ROLLOUT_LOG_DIR="${DREAM_ROLLOUT_LOG_DIR:-${log_dir}/rollout_debug}"
# Each DP rank writes to rollout_debug/rank{N}.rollout.log (set DREAM_ROLLOUT_LOG_RANK=N to restrict).
unset DREAM_ROLLOUT_LOG_RANK
val_generations_dir="${log_dir}/val_generations"
mkdir -p "${DREAM_ROLLOUT_LOG_DIR}" "${val_generations_dir}"
echo "[INFO] Rollout debug: DREAM_ROLLOUT_VERBOSE=${DREAM_ROLLOUT_VERBOSE} log_dir=${DREAM_ROLLOUT_LOG_DIR} (per-rank rank0..rank$((n_gpus_per_node - 1)).rollout.log)"
if [ "${smoke_test}" -eq 1 ]; then
  echo "[INFO] Smoke val: val_before_train=True test_freq=1 steps=1"
  echo "[INFO] HumanEval dumps: ${val_generations_dir}/0.jsonl (pre-train) -> ${val_generations_dir}/1.jsonl (post step1)"
fi
echo "[INFO] WANDB_MODE=${WANDB_MODE} project=${WANDB_PROJECT} WANDB_DIR=${WANDB_DIR:-${log_dir}/wandb}"

sglang_extra_args=()
if [ "$engine" = "sglang" ]; then
  sglang_extra_args+=(
    "+actor_rollout_ref.rollout.dllm_algorithm=FullAttnMultiBlock"
    "+actor_rollout_ref.rollout.attention_backend=${sglang_attention_backend}"
    "+actor_rollout_ref.rollout.disable_cuda_graph=${sglang_disable_cuda_graph}"
    "+actor_rollout_ref.rollout.mem_fraction_static=${sglang_mem_fraction_static}"
    "+actor_rollout_ref.rollout.max_running_requests=1"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${sglang_gpu_memory_utilization}"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.enforce_eager=True"
  )
fi

# Per-run WANDB_DIR; force cloud sync when logger includes wandb (do not inherit offline).
export WANDB_DIR="${log_dir}/wandb"
mkdir -p "${WANDB_DIR}"
if [[ "${trainer_logger}" == *wandb* ]]; then
  export WANDB_MODE=online
fi

echo "[INFO] PYTHON=${PYTHON}"
"${PYTHON}" -m verl.trainer.dllm_main_ppo \
  algorithm.adv_estimator=grpo \
  +algorithm.name=${algorithm} \
  reward_model.reward_manager=dllm \
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
  +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
  +reward_model.reward_kwargs.enable_tpf_efficiency=False \
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
  +actor_rollout_ref.actor.enable_cfl=${enable_cfl} \
  +actor_rollout_ref.actor.cfl_coef=${cfl_coef} \
  +actor_rollout_ref.actor.cfl_temperature=${cfl_temperature} \
  +actor_rollout_ref.actor.cfl_gate_passed_only=${cfl_gate_passed_only} \
  +actor_rollout_ref.actor.cfl_gate_positive_adv_only=${cfl_gate_positive_adv_only} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing:-False} \
  ++actor_rollout_ref.model.enable_activation_offload=${enable_activation_offload} \
  actor_rollout_ref.model.trust_remote_code=True \
  +actor_rollout_ref.model.attn_implementation="flash_attention_2" \
  +actor_rollout_ref.model.baseline=${baseline} \
  actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload} \
  +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[DreamDecoderLayer] \
  +actor_rollout_ref.actor.mc_num=${mc_num} \
  +actor_rollout_ref.actor.n_l=${n_l} \
  +actor_rollout_ref.actor.cfg_scale=0.0 \
  +actor_rollout_ref.actor.baseline=${baseline} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=${engine} \
  +actor_rollout_ref.rollout.use_cache=True \
  +actor_rollout_ref.rollout.dual_cache=False \
  +actor_rollout_ref.rollout.dllm_decode=multiblock \
  +actor_rollout_ref.rollout.d3llm_threshold=0.5 \
  +actor_rollout_ref.rollout.d3llm_block_add_threshold=0.1 \
  +actor_rollout_ref.rollout.d3llm_decoded_token_threshold=0.95 \
  +actor_rollout_ref.rollout.d3llm_cache_delay_iter=32 \
  +actor_rollout_ref.rollout.d3llm_early_stop=True \
  +actor_rollout_ref.rollout.mask_token_id=${mask_token_id} \
  +actor_rollout_ref.rollout.per_sample_seed=True \
  +actor_rollout_ref.rollout.rollout_verbose=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=${sglang_gpu_memory_utilization:-0.4} \
  actor_rollout_ref.rollout.n=${n_rollout} \
  actor_rollout_ref.rollout.temperature=${train_temperature} \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
  +actor_rollout_ref.rollout.val_kwargs.num_diffusion_steps=${val_num_diffusion_steps} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
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
  +trainer.val_only=False \
  trainer.n_gpus_per_node=${n_gpus_per_node} \
  trainer.nnodes=1 \
  trainer.default_local_dir=${ckpt_dir} \
  trainer.save_freq=${save_freq} \
  trainer.test_freq=${test_freq} \
  trainer.total_epochs=${total_epoch} \
  +trainer.validation_data_dir="${val_generations_dir}" \
  ${smoke_total_training_steps:+trainer.total_training_steps=${smoke_total_training_steps}} \
  custom_reward_function.path="verl/utils/reward_score/__init__.py" \
  custom_reward_function.name="dllm_rm" \
  "${sglang_extra_args[@]}" \
  2>&1 | tee "${log_dir}/${exp_name}.log"
