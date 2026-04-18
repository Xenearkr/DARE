#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage: bash recipe/llada/run_d1_llada_8b_instruct_multi_node_user.sh [options]

Options:
  --task {math|code|sudoku|countdown}
  --nnodes N
  --node_rank R
  --head_address HOST[:PORT]
  --nproc_per_node N
  --model_path PATH
  --resume_path PATH
  --ckpt_dir DIR
  --help

Notes:
  - Users can pass --nnodes/--node_rank explicitly, or rely on scheduler env vars such
    as NODE_RANK/GROUP_RANK/RJOB_TASK_INDEX and NNODES/SLURM_NNODES.
  - Worker nodes can discover the Ray head from --head_address, RAY_HEAD_ADDRESS,
    MASTER_ADDR, COORDINATOR_ADDR, or the shared rendezvous file.
  - Override HF_HOME and CHECKPOINT_ROOT if you want to use a custom shared cache
    or checkpoint directory.
EOF
}

get_node_ip() {
    local candidate=""

    for candidate in \
        "${SOCKET_IP:-}" \
        "${MY_HOST_IP:-}" \
        "${POD_IP:-}" \
        "${HOST_IP:-}" \
        "${POD_IPV4:-}"; do
        if [[ -n "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    candidate="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
    if [[ -n "${candidate}" ]]; then
        echo "${candidate}"
        return 0
    fi

    candidate="$(
        python3 - <<'PY'
import socket

try:
    print(socket.gethostbyname(socket.gethostname()))
except Exception:
    print("")
PY
    )"
    if [[ -n "${candidate}" ]]; then
        echo "${candidate}"
        return 0
    fi

    echo "127.0.0.1"
}

first_defined_int() {
    local var_name
    local value=""
    for var_name in "$@"; do
        value="${!var_name:-}"
        if [[ "${value}" =~ ^[0-9]+$ ]]; then
            echo "${value}"
            return 0
        fi
    done
    return 1
}

first_defined_string() {
    local var_name
    local value=""
    for var_name in "$@"; do
        value="${!var_name:-}"
        if [[ -n "${value}" ]]; then
            echo "${value}"
            return 0
        fi
    done
    return 1
}

resolve_node_rank() {
    local explicit_rank="${1:-}"

    if [[ -n "${explicit_rank}" ]]; then
        echo "${explicit_rank}"
        return 0
    fi

    first_defined_int \
        NODE_RANK \
        GROUP_RANK \
        SLURM_NODEID \
        PMI_NODE_RANK \
        OMPI_COMM_WORLD_NODE_RANK \
        RJOB_TASK_INDEX || echo 0
}

resolve_nnodes() {
    local explicit_nnodes="${1:-}"

    if [[ -n "${explicit_nnodes}" ]]; then
        echo "${explicit_nnodes}"
        return 0
    fi

    first_defined_int \
        NNODES \
        SLURM_NNODES \
        SLURM_JOB_NUM_NODES \
        PBS_NUM_NODES \
        PET_NNODES || echo 1
}

normalize_head_address() {
    local address="${1:-}"

    if [[ -z "${address}" ]]; then
        echo ""
    elif [[ "${address}" == *:* ]]; then
        echo "${address}"
    else
        echo "${address}:${ray_port}"
    fi
}

resolve_head_address() {
    local explicit_head_address="${1:-}"

    if [[ -n "${explicit_head_address}" ]]; then
        normalize_head_address "${explicit_head_address}"
        return 0
    fi

    normalize_head_address "$(first_defined_string RAY_HEAD_ADDRESS MASTER_ADDR COORDINATOR_ADDR || true)"
}

resolve_multi_node_run_id() {
    local resolved_nnodes="$1"
    local scheduler_job_id=""
    local manual_key=""

    if [[ -n "${MULTI_NODE_RUN_ID:-}" ]]; then
        echo "${MULTI_NODE_RUN_ID}"
        return 0
    fi

    scheduler_job_id="$(first_defined_string JOB_ID SLURM_JOB_ID PBS_JOBID LSB_JOBID COBALT_JOBID RJOB_JOB_ID || true)"
    if [[ -n "${scheduler_job_id}" ]]; then
        echo "${scheduler_job_id}"
        return 0
    fi

    manual_key="$(first_defined_string MASTER_ADDR COORDINATOR_ADDR RAY_HEAD_ADDRESS || true)"
    manual_key="${manual_key//[^A-Za-z0-9._-]/_}"
    if [[ -n "${manual_key}" ]]; then
        echo "manual-${USER:-user}-nnodes${resolved_nnodes}-${manual_key}"
        return 0
    fi

    echo "manual-${USER:-user}-nnodes${resolved_nnodes}"
}

get_rendezvous_file() {
    echo "${ray_rendezvous_dir}/${multi_node_run_id}.ray_head"
}

publish_head_address() {
    local address="$1"
    local rendezvous_file
    local tmp_file

    rendezvous_file="$(get_rendezvous_file)"
    tmp_file="${rendezvous_file}.tmp.$$"
    mkdir -p "${ray_rendezvous_dir}"
    printf '%s\n' "${address}" > "${tmp_file}"
    mv -f "${tmp_file}" "${rendezvous_file}"
}

wait_for_head_address() {
    local max_wait_seconds="${1:-300}"
    local waited=0
    local rendezvous_file
    local detected_head=""

    rendezvous_file="$(get_rendezvous_file)"

    while (( waited < max_wait_seconds )); do
        if [[ -s "${rendezvous_file}" ]]; then
            detected_head="$(head -n 1 "${rendezvous_file}")"
            detected_head="$(normalize_head_address "${detected_head}")"
            if [[ -n "${detected_head}" ]]; then
                echo "${detected_head}"
                return 0
            fi
        fi

        sleep 5
        waited=$((waited + 5))
    done

    return 1
}

wait_for_ray_cluster() {
    local expected_nodes="$1"
    local max_wait_seconds="${2:-300}"
    local waited=0
    local alive_nodes=0

    while (( waited < max_wait_seconds )); do
        alive_nodes="$(
            python3 - <<'PY'
import os
import ray

address = os.environ["RAY_ADDRESS"]

try:
    ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")
    print(sum(1 for node in ray.nodes() if node.get("Alive")))
except Exception:
    print(0)
finally:
    try:
        ray.shutdown()
    except Exception:
        pass
PY
        )"

        if [[ "${alive_nodes}" =~ ^[0-9]+$ ]] && (( alive_nodes >= expected_nodes )); then
            return 0
        fi

        sleep 10
        waited=$((waited + 10))
    done

    echo "[ERROR] Ray cluster did not reach ${expected_nodes} alive nodes within ${max_wait_seconds} seconds"
    return 1
}

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-DARE}"
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_MODE="${WANDB_MODE:-offline}"
default_cache_root="${XDG_CACHE_HOME:-${HOME:-${REPO_ROOT}}/.cache}"
export HF_HOME="${HF_HOME:-${default_cache_root}/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
checkpoint_root="${CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/checkpoints}"

NODE_IP="$(get_node_ip)"
export MY_HOST_IP="${MY_HOST_IP:-${NODE_IP}}"
export HOST_IP="${HOST_IP:-${NODE_IP}}"

if [[ -n "${NCCL_SOCKET_IFNAME:-}" && -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
    export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
fi
if [[ -z "${NCCL_IB_GID_INDEX:-}" && -n "${NVSHMEM_IB_GID_INDEX:-}" ]]; then
    export NCCL_IB_GID_INDEX="${NVSHMEM_IB_GID_INDEX}"
fi
if [[ -z "${NCCL_IB_HCA:-}" && -n "${NVSHMEM_HCA_LIST:-}" ]]; then
    export NCCL_IB_HCA="${NVSHMEM_HCA_LIST}"
fi

model="llada"
model_path="models/LLaDA-8B-Instruct"
task="math"
algorithm="d1"
engine="hf"
nnodes=""
node_rank=""
head_address=""
nproc_per_node="${NPROC_PER_NODE:-${PROC_PER_NODE:-8}}"
resume_path=""
ckpt_dir=""
ray_port="${RAY_PORT:-6379}"
dashboard_port="${RAY_DASHBOARD_PORT:-8265}"
ray_cluster_wait_seconds="${RAY_CLUSTER_WAIT_SECONDS:-300}"
ray_head_wait_seconds="${RAY_HEAD_WAIT_SECONDS:-300}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model="$2"
            shift 2
            ;;
        --model_path)
            model_path="$2"
            shift 2
            ;;
        --task)
            task="$2"
            shift 2
            ;;
        --algorithm)
            algorithm="$2"
            shift 2
            ;;
        --engine)
            engine="$2"
            shift 2
            ;;
        --nnodes)
            nnodes="$2"
            shift 2
            ;;
        --node_rank)
            node_rank="$2"
            shift 2
            ;;
        --head_address)
            head_address="$2"
            shift 2
            ;;
        --nproc_per_node)
            nproc_per_node="$2"
            shift 2
            ;;
        --resume_path)
            resume_path="$2"
            shift 2
            ;;
        --ckpt_dir)
            ckpt_dir="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

NODE_RANK="$(resolve_node_rank "${node_rank}")"
NNODES="$(resolve_nnodes "${nnodes}")"
head_address="$(resolve_head_address "${head_address}")"
multi_node_run_id="$(resolve_multi_node_run_id "${NNODES}")"
ray_rendezvous_dir="${RAY_RENDEZVOUS_DIR:-${REPO_ROOT}/logs/ray_rendezvous}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((nproc_per_node - 1)))"
fi

n_gpus_per_node="$(echo "${CUDA_VISIBLE_DEVICES}" | tr "," "\n" | wc -l | tr -d ' ')"
total_gpus=$((NNODES * n_gpus_per_node))

default_ulysses_sequence_parallel_size=1
if (( total_gpus >= 2 )); then
    default_ulysses_sequence_parallel_size=2
fi

batch_size="${BATCH_SIZE:-${total_gpus}}"
n_rollout="${N_ROLLOUT:-8}"
lr="${LR:-5e-7}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
train_temperature="${TRAIN_TEMPERATURE:-0.6}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-${default_ulysses_sequence_parallel_size}}"
rollout_tensor_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
mc_num="${MC_NUM:-1}"
n_l="${N_L:-1}"
block_length="${BLOCK_LENGTH:-32}"

if [[ ! " llada " =~ " ${model} " ]]; then
    echo "Error: Invalid model '${model}'"
    exit 1
fi

if [[ ! " d1 " =~ " ${algorithm} " ]]; then
    echo "Error: Invalid algorithm '${algorithm}'"
    exit 1
fi

if [[ ! " hf " =~ " ${engine} " ]]; then
    echo "Error: Invalid engine '${engine}'"
    exit 1
fi

valid_tasks=("math" "code" "sudoku" "countdown")
if [[ ! " ${valid_tasks[*]} " =~ " ${task} " ]]; then
    echo "Error: Invalid task '${task}'"
    echo "Supported tasks: ${valid_tasks[*]}"
    exit 1
fi

if [[ "${task}" == "math" ]]; then
    train_files="['data/preprocessed/rl/train/math_1.parquet','data/preprocessed/rl/train/gsm8k_1.parquet']"
    val_files="['data/preprocessed/rl/test/math500_1.parquet','data/preprocessed/rl/test/gsm8k_1.parquet']"
    max_prompt_length=512
    max_response_length=512
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
elif [[ "${task}" == "code" ]]; then
    train_files="['data/preprocessed/rl/train/lcbv5-K8_1.parquet','data/preprocessed/rl/train/primeintellect-K8_1.parquet','data/preprocessed/rl/train/taco-K8_1.parquet']"
    val_files="['data/preprocessed/rl/test/mbpp_1.parquet','data/preprocessed/rl/test/humaneval_1.parquet','data/preprocessed/rl/test/humanevalplus_1.parquet']"
    max_prompt_length=1024
    max_response_length=512
    num_diffusion_steps="${max_response_length}"
    total_epoch=5
elif [[ "${task}" == "countdown" ]]; then
    train_files="['data/preprocessed/rl/train/countdown-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/countdown_1.parquet']"
    max_prompt_length=512
    max_response_length=256
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
else
    train_files="['data/preprocessed/rl/train/sudoku-n20000_1.parquet']"
    val_files="['data/preprocessed/rl/test/sudoku_1.parquet']"
    max_prompt_length=512
    max_response_length=256
    num_diffusion_steps=$((max_response_length / 2))
    total_epoch=1
fi

val_num_diffusion_steps="${max_response_length}"
baseline="${model}-${task}-${algorithm}-${engine}"
timestamp="$(date +"%Y%m%d_%H%M%S")"
project_name="${WANDB_PROJECT}"
exp_name="${baseline}-bsz${batch_size}-n${n_rollout}-prompt${max_prompt_length}-response${max_response_length}-step${num_diffusion_steps}-lr${lr}-temp${train_temperature}-n_l${n_l}-mc_num${mc_num}-nodes${NNODES}-gpu${n_gpus_per_node}-${timestamp}"
resolved_ckpt_dir="${ckpt_dir:-${checkpoint_root}/${project_name}/${exp_name}}"
log_dir="${REPO_ROOT}/logs/${project_name}/${exp_name}"

mkdir -p "${resolved_ckpt_dir}" "${log_dir}"

resume_args=("trainer.default_local_dir=${resolved_ckpt_dir}")
if [[ -n "${resume_path}" ]]; then
    resume_args=(
        "trainer.resume_mode=resume_path"
        "trainer.resume_from_path=${resume_path}"
        "trainer.default_local_dir=${resolved_ckpt_dir}"
    )
fi

echo "[INFO] Launching ${baseline}: node_rank=${NODE_RANK}, nnodes=${NNODES}, node_ip=${NODE_IP}"
echo "[INFO] Logs: ${log_dir}"
echo "[INFO] Checkpoints: ${resolved_ckpt_dir}"

if [[ "${NNODES}" -eq 1 ]]; then
    ray stop --force >/dev/null 2>&1 || true
    rm -rf /tmp/ray || true
fi

if [[ "${NNODES}" -gt 1 ]]; then
    if [[ "${NODE_RANK}" == "0" ]]; then
        export RAY_HEAD_ADDRESS
        RAY_HEAD_ADDRESS="$(normalize_head_address "${head_address:-${NODE_IP}}")"
        export RAY_ADDRESS="${RAY_HEAD_ADDRESS}"
        publish_head_address "${RAY_HEAD_ADDRESS}"

        echo "[INFO] Starting Ray head on ${RAY_HEAD_ADDRESS}"
        ray start \
            --head \
            --node-ip-address="${NODE_IP}" \
            --dashboard-host=0.0.0.0 \
            --port="${ray_port}" \
            --dashboard-port="${dashboard_port}" \
            --block &
        RAY_PID=$!
        sleep 10
    else
        if [[ -z "${head_address}" ]]; then
            head_address="$(wait_for_head_address "${ray_head_wait_seconds}")"
        fi

        if [[ -z "${head_address}" ]]; then
            echo "[ERROR] Could not determine the Ray head address for worker node ${NODE_RANK}"
            exit 1
        fi

        export RAY_HEAD_ADDRESS="${head_address}"
        export RAY_ADDRESS="${RAY_HEAD_ADDRESS}"

        echo "[INFO] Connecting Ray worker to ${RAY_HEAD_ADDRESS}"
        ray start \
            --address="${RAY_HEAD_ADDRESS}" \
            --node-ip-address="${NODE_IP}" \
            --block &
        RAY_PID=$!
        sleep 10
    fi
else
    export RAY_HEAD_ADDRESS="127.0.0.1:${ray_port}"
    export RAY_ADDRESS="${RAY_HEAD_ADDRESS}"

    echo "[INFO] Starting single-node Ray head on ${RAY_HEAD_ADDRESS}"
    ray start \
        --head \
        --node-ip-address=127.0.0.1 \
        --dashboard-host=0.0.0.0 \
        --port="${ray_port}" \
        --dashboard-port="${dashboard_port}" \
        --block &
    RAY_PID=$!
    sleep 10
fi

if [[ "${NODE_RANK}" == "0" || "${NNODES}" -eq 1 ]]; then
    if [[ "${NNODES}" -gt 1 ]]; then
        wait_for_ray_cluster "${NNODES}" "${ray_cluster_wait_seconds}"
    fi

    train_cmd=(
        python3
        -u
        -m
        verl.trainer.dllm_main_ppo
        algorithm.adv_estimator=grpo
        "+algorithm.name=${algorithm}"
        reward_model.reward_manager=dllm
        "+reward_model.reward_kwargs.overlong_buffer_cfg.enable=False"
        "+reward_model.reward_kwargs.max_resp_len=${max_response_length}"
        "data.train_files=${train_files}"
        "data.val_files=${val_files}"
        "data.train_batch_size=${batch_size}"
        data.val_batch_size=64
        "data.max_prompt_length=${max_prompt_length}"
        "data.max_response_length=${max_response_length}"
        data.filter_overlong_prompts=True
        data.truncation=error
        "+actor_rollout_ref.algorithm.name=${algorithm}"
        "+actor_rollout_ref.model.name=${model}"
        "actor_rollout_ref.model.path=${model_path}"
        "actor_rollout_ref.actor.optim.lr=${lr}"
        actor_rollout_ref.actor.optim.weight_decay=0.01
        actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.actor.strategy=fsdp2
        "actor_rollout_ref.actor.ppo_mini_batch_size=${batch_size}"
        actor_rollout_ref.actor.use_dynamic_bsz=True
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120
        actor_rollout_ref.actor.use_kl_loss=False
        actor_rollout_ref.actor.kl_loss_coef=0.0
        actor_rollout_ref.actor.kl_loss_type=low_var_kl
        actor_rollout_ref.actor.entropy_coeff=0.0
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}"
        actor_rollout_ref.actor.loss_agg_mode=token-mean
        actor_rollout_ref.model.enable_gradient_checkpointing=False
        actor_rollout_ref.model.trust_remote_code=True
        +actor_rollout_ref.model.attn_implementation=flash_attention_2
        "+actor_rollout_ref.model.baseline=${baseline}"
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
        +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
        +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bfloat16
        +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bfloat16
        +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bfloat16
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size}"
        +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock]
        "+actor_rollout_ref.actor.mc_num=${mc_num}"
        "+actor_rollout_ref.actor.n_l=${n_l}"
        +actor_rollout_ref.actor.cfg_scale=0.0
        "+actor_rollout_ref.actor.baseline=${baseline}"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tensor_parallel_size}"
        "actor_rollout_ref.rollout.name=${engine}"
        +actor_rollout_ref.rollout.use_cache=True
        +actor_rollout_ref.rollout.dual_cache=False
        "actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}"
        "actor_rollout_ref.rollout.n=${n_rollout}"
        "actor_rollout_ref.rollout.temperature=${train_temperature}"
        actor_rollout_ref.rollout.do_sample=True
        actor_rollout_ref.rollout.val_kwargs.do_sample=True
        actor_rollout_ref.rollout.val_kwargs.n=1
        actor_rollout_ref.rollout.val_kwargs.temperature=0.0
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95
        "+actor_rollout_ref.rollout.val_kwargs.num_diffusion_steps=${val_num_diffusion_steps}"
        actor_rollout_ref.rollout.max_num_batched_tokens=11000
        actor_rollout_ref.rollout.enable_chunked_prefill=True
        "+actor_rollout_ref.rollout.num_diffusion_steps=${num_diffusion_steps}"
        "+actor_rollout_ref.rollout.block_length=${block_length}"
        "+actor_rollout_ref.rollout.mc_num=${mc_num}"
        "+actor_rollout_ref.rollout.n_l=${n_l}"
        +actor_rollout_ref.rollout.cfg_scale=0.0
        actor_rollout_ref.ref.fsdp_config.param_offload=True
        algorithm.use_kl_in_reward=False
        trainer.critic_warmup=0
        'trainer.logger=["console","wandb"]'
        "trainer.project_name=${project_name}"
        "trainer.experiment_name=${exp_name}"
        trainer.val_before_train=False
        "trainer.n_gpus_per_node=${n_gpus_per_node}"
        "trainer.nnodes=${NNODES}"
    )
    train_cmd+=("${resume_args[@]}")
    train_cmd+=(
        trainer.save_freq=100
        trainer.test_freq=10
        "trainer.total_epochs=${total_epoch}"
        custom_reward_function.path=verl/utils/reward_score/__init__.py
        custom_reward_function.name=dllm_rm
    )

    "${train_cmd[@]}" >> "${log_dir}/${baseline}-${timestamp}.out" 2>> "${log_dir}/${baseline}-${timestamp}.err"
else
    wait "${RAY_PID}"
fi
