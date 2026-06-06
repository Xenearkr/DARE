#!/bin/bash
# BGPO / bgpo-cj on EvalPlus HumanEval+MBPP (direct overfit sanity).
# bgpo-cj: only forward_process uses AR-prefix suffix mask; actor/rollout/eval same as BGPO.
#
# Usage:
#   bash recipe/dream/run_bgpo_dream_coder_evalplus_direct.sh --smoke --algorithm bgpo-cj --engine sglang
#   bash recipe/dream/run_bgpo_dream_coder_evalplus_direct.sh --smoke --algorithm bgpo --engine sglang
#   bash recipe/dream/run_bgpo_dream_coder_evalplus_direct.sh --engine sglang
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

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
DARE_SUPPRESS_WARNINGS="ignore:pkg_resources is deprecated:UserWarning,ignore:The pynvml package is deprecated:FutureWarning"
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}${DARE_SUPPRESS_WARNINGS}"

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
val_only=0
model_path=""
algorithm=bgpo
engine=sglang

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) model_path="$2"; shift 2 ;;
    --engine) engine="$2"; shift 2 ;;
    --algorithm) algorithm="$2"; shift 2 ;;
    --smoke) smoke_test=1; shift ;;
    --val-only) val_only=1; shift ;;
    *) echo "[WARN] Unknown arg: $1"; shift ;;
  esac
done

valid_algorithms=(bgpo bgpo-cj)
if [[ ! " ${valid_algorithms[*]} " =~ " ${algorithm} " ]]; then
  echo "[ERROR] Invalid algorithm '${algorithm}'. Supported: ${valid_algorithms[*]}"
  exit 1
fi

valid_engines=(hf sglang)
if [[ ! " ${valid_engines[*]} " =~ " ${engine} " ]]; then
  echo "[ERROR] Invalid engine '${engine}'. Supported: ${valid_engines[*]}"
  exit 1
fi

model=dream
model_path=${model_path:-models/finetune_d3LLM}
task=code
baseline="${model}-${task}-d3llm-${algorithm}-${engine}-evalplus-direct"

mask_token_id=151666
pad_token_id=151643
block_length=32

sglang_mem_fraction_static=0.45
sglang_gpu_memory_utilization=0.4
sglang_attention_backend=fa3
sglang_disable_cuda_graph=False
actor_param_offload=False
actor_optimizer_offload=False
enable_activation_offload=False

HE_EVALPLUS="data/preprocessed/rl/test/humaneval_evalplus_1.parquet"
MBPP_EVALPLUS="data/preprocessed/rl/test/mbpp_evalplus_1.parquet"
EVALPLUS_TRAIN_FILES="['${HE_EVALPLUS}','${MBPP_EVALPLUS}']"

ensure_evalplus_parquets() {
  if [[ -f "${HE_EVALPLUS}" && -f "${MBPP_EVALPLUS}" ]]; then
    return 0
  fi
  echo "[INFO] Building EvalPlus parquets: ${PYTHON} recipe/d3llm/build_evalplus_code_mix.py --skip-train"
  "${PYTHON}" recipe/d3llm/build_evalplus_code_mix.py --skip-train --skip-tokenizer-check || {
    echo "[ERROR] Failed to build EvalPlus parquets."
    exit 1
  }
}

build_smoke_subset() {
  local src="$1"
  local dst="$2"
  local n="$3"
  if [[ -f "${dst}" ]]; then
    return 0
  fi
  echo "[INFO] Building smoke subset (${n} samples): ${dst}"
  "${PYTHON}" - <<PY
import pandas as pd
df = pd.read_parquet("${src}").head(${n})
extra = df["extra_info"].apply(lambda x: {**(x or {}), "task": (x or {}).get("task") or "code"})
df = df.copy()
df["extra_info"] = extra
df.to_parquet("${dst}", index=False)
print(f"wrote ${dst}")
PY
}

# Always 512 — aligned with d3LLM evalplus max_new_tokens.
max_response_length=512
max_prompt_length=1024

if [ "${smoke_test}" -eq 1 ]; then
  export CUDA_VISIBLE_DEVICES="${DARE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  train_files="${EVALPLUS_TRAIN_FILES}"

  HE_SMOKE_VAL="data/preprocessed/rl/test/humaneval_evalplus_smoke_8.parquet"
  MBPP_SMOKE_VAL="data/preprocessed/rl/test/mbpp_evalplus_smoke_8.parquet"
  build_smoke_subset "${HE_EVALPLUS}" "${HE_SMOKE_VAL}" 8
  build_smoke_subset "${MBPP_EVALPLUS}" "${MBPP_SMOKE_VAL}" 8
  val_files="['${HE_SMOKE_VAL}','${MBPP_SMOKE_VAL}']"

  batch_size=4
  n_rollout=4
  mc_num=4
  n_l=4
  ppo_max_token_len_per_gpu=2048
  max_num_batched_tokens=4096
  val_batch_size=32
  save_freq=0
  test_freq=1
  val_before_train=True
  total_epoch=1
  trainer_logger='["console"]'
  enable_gradient_checkpointing=False
  smoke_total_training_steps=1
  log_val_generations=8
  export WANDB_MODE=offline
  train_temperature=0.4
  val_temperature=0.0
  val_do_sample=False
else
  train_files="${EVALPLUS_TRAIN_FILES}"
  val_files="${HE_EVALPLUS}"
  batch_size=8
  n_rollout=4
  mc_num=8
  n_l=8
  ppo_max_token_len_per_gpu=3072
  max_num_batched_tokens=6144
  val_batch_size=32
  save_freq=40
  test_freq=10
  val_before_train=True
  total_epoch=4
  trainer_logger='["console","wandb"]'
  enable_gradient_checkpointing=True
  train_temperature=0.4
  val_temperature=0.0
  val_do_sample=False
fi

if [ "${val_only}" -eq 1 ]; then
  val_files="${HE_EVALPLUS}"
  train_files="${HE_EVALPLUS}"
  batch_size=4
  n_rollout=1
  mc_num=4
  n_l=4
  val_batch_size=32
  val_before_train=True
  total_epoch=1
  trainer_logger='["console"]'
  enable_gradient_checkpointing=False
  export WANDB_MODE=offline
  log_val_generations=164
  echo "[INFO] Val-only: 164 HumanEval EvalPlus, no training"
fi

if [ "$engine" = "sglang" ]; then
  sglang_mem_fraction_static=0.32
  sglang_gpu_memory_utilization=0.32
  sglang_attention_backend=torch_native
  sglang_disable_cuda_graph=True
  actor_param_offload=True
  actor_optimizer_offload=True
  enable_activation_offload=True
fi

log_val_generations=${log_val_generations:-0}

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
train_temperature="${train_temperature:-0.4}"
val_temperature="${val_temperature:-0.0}"
val_top_p="${val_top_p:-1.0}"
val_do_sample="${val_do_sample:-False}"

enable_cfl=True
cfl_coef=0.01
cfl_temperature=0.5
cfl_gate_positive_adv_only=False
if [ "${smoke_test}" -eq 1 ]; then
  cfl_gate_passed_only=False
else
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

ensure_evalplus_parquets

if [ "${algorithm}" = "bgpo-cj" ]; then
  echo "[INFO] bgpo-cj: AR-prefix suffix mask in forward_process (same actor/ELBO as BGPO)"
fi
echo "[INFO] engine=${engine} smoke=${smoke_test} GPUs=${n_gpus_per_node} train_temperature=${train_temperature}"
echo "[INFO] Direct EvalPlus train: ${HE_EVALPLUS} + ${MBPP_EVALPLUS} (542 samples)"
echo "[INFO] max_response_length=${max_response_length} (train+val, aligned with d3LLM evalplus)"
echo "[INFO] CFL: enable=${enable_cfl} coef=${cfl_coef} temperature=${cfl_temperature} gate_passed=${cfl_gate_passed_only}"
echo "[INFO] W&B metrics: val-core/humaneval/acc/mean@1, val-core/mbpp/acc/mean@1"
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
export DREAM_ROLLOUT_VERBOSE="${DREAM_ROLLOUT_VERBOSE:-1}"
export DREAM_ROLLOUT_LOG_DIR="${DREAM_ROLLOUT_LOG_DIR:-${log_dir}/rollout_debug}"
unset DREAM_ROLLOUT_LOG_RANK
val_generations_dir="${log_dir}/val_generations"
mkdir -p "${DREAM_ROLLOUT_LOG_DIR}" "${val_generations_dir}"
echo "[INFO] Rollout debug: DREAM_ROLLOUT_VERBOSE=${DREAM_ROLLOUT_VERBOSE} log_dir=${DREAM_ROLLOUT_LOG_DIR}"
if [ "${smoke_test}" -eq 1 ]; then
  echo "[INFO] Smoke: val_before_train=${val_before_train} test_freq=${test_freq} steps=${smoke_total_training_steps:-epoch}"
  echo "[INFO] Val dumps: ${val_generations_dir}/1.jsonl (post step1, no pre-train val)"
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

export WANDB_DIR="${log_dir}/wandb"
mkdir -p "${WANDB_DIR}"
if [[ "${trainer_logger}" == *wandb* && "${WANDB_MODE:-online}" != "offline" ]]; then
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
  +trainer.val_only=$([ "${val_only}" -eq 1 ] && echo True || echo False) \
  trainer.n_gpus_per_node=${n_gpus_per_node} \
  trainer.nnodes=1 \
  trainer.default_local_dir=${ckpt_dir} \
  trainer.save_freq=${save_freq} \
  trainer.test_freq=${test_freq} \
  trainer.total_epochs=${total_epoch} \
  trainer.log_val_generations=${log_val_generations} \
  +trainer.validation_data_dir="${val_generations_dir}" \
  ${smoke_total_training_steps:+trainer.total_training_steps=${smoke_total_training_steps}} \
  custom_reward_function.path="verl/utils/reward_score/__init__.py" \
  custom_reward_function.name="dllm_rm" \
  "${sglang_extra_args[@]}" \
  2>&1 | tee "${log_dir}/${exp_name}.log"
