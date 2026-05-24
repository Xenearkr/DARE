#!/usr/bin/env bash
# DARE 一键环境配置
# 适配本机: Ubuntu 24.04, 4×RTX A6000, CUDA 12.x, conda @ ~/anaconda3
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/u-liujc/anaconda3}"
TRAIN_ENV="${TRAIN_ENV:-DARE}"
EVAL_ENV="${EVAL_ENV:-opencompass}"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
SGLANG_DIR="${ROOT}/third_party/sglang"
HUMAN_EVAL_DIR="${ROOT}/opencompass/human-eval"
ENV_FILE="${ROOT}/scripts/dare_env.sh"

SETUP_TRAIN=1
SETUP_EVAL=1
SETUP_SGLANG=1
FLASH_ATTN_ONLY=0

usage() {
  cat <<'EOF'
用法: bash setup_dare_env.sh [选项]
  --train-only      仅配置训练环境 (conda: DARE)
  --eval-only       仅配置评测环境 (conda: opencompass)
  --flash-attn-only 仅安装/重编译 flash-attn（需已有 DARE 环境）
  --no-sglang       跳过 SGLang 源码安装
  -h, --help        显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-only) SETUP_EVAL=0; SETUP_SGLANG=0 ;;
    --eval-only)  SETUP_TRAIN=0; SETUP_SGLANG=0 ;;
    --flash-attn-only) FLASH_ATTN_ONLY=1; SETUP_EVAL=0; SETUP_SGLANG=0 ;;
    --no-sglang)  SETUP_SGLANG=0 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "未知参数: $1"; usage; exit 1 ;;
  esac
  shift
done

[[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]] || {
  echo "[ERROR] 未找到 conda: ${CONDA_ROOT}"
  exit 1
}
# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

conda_env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

ensure_env() {
  local name="$1" py="$2"
  if conda_env_exists "$name"; then
    echo "[INFO] conda 环境已存在: ${name}"
  else
    echo "[INFO] 创建 conda 环境: ${name} (python=${py})"
    conda create -n "$name" "python=${py}" -y
  fi
}

pip_in() {
  local env="$1"
  shift
  local pip="${CONDA_ROOT}/envs/${env}/bin/pip"
  [[ -x "$pip" ]] || { echo "[ERROR] 未找到 ${pip}"; exit 1; }
  "$pip" "$@"
}

# flash-attn 需 nvcc + CUDA_HOME；torch 2.9 与官方 torch2.8 wheel ABI 不兼容，必须源码编译
ensure_cuda_toolkit() {
  local env="$1"
  local prefix="${CONDA_ROOT}/envs/${env}"
  if [[ -x "${prefix}/bin/nvcc" ]]; then
    echo "[INFO] nvcc 已存在: ${prefix}/bin/nvcc"
    return 0
  fi
  echo "[INFO] 通过 conda 安装 CUDA 编译工具链 (nvcc 12.9)..."
  conda install -n "$env" -y -c nvidia cuda-nvcc=12.9 cuda-cccl cuda-cudart-dev
}

install_flash_attn() {
  local env="$1"
  local prefix="${CONDA_ROOT}/envs/${env}"

  if conda run -n "$env" python -c "import flash_attn" 2>/dev/null; then
    echo "[INFO] flash-attn 已安装，跳过"
    return 0
  fi

  # 若曾装过 torch2.8 预编译 wheel，与 torch 2.9 不兼容，需先卸载
  if conda run -n "$env" pip show flash-attn &>/dev/null; then
    echo "[WARN] 检测到 flash-attn 但无法 import，卸载后重新编译..."
    conda run -n "$env" pip uninstall -y flash-attn
  fi

  ensure_cuda_toolkit "$env"

  echo "[INFO] 从源码编译 flash-attn==2.8.3（torch 2.9 无匹配 wheel，约 15–40 分钟）..."
  echo "[INFO] CUDA_HOME=${prefix}"
  MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
    conda run -n "$env" env \
      CUDA_HOME="${prefix}" \
      PATH="${prefix}/bin:${PATH}" \
      MAX_JOBS="${MAX_JOBS}" \
      pip install flash-attn==2.8.3 --no-build-isolation

  conda run -n "$env" python -c "import flash_attn; print('[OK] flash_attn', flash_attn.__version__)"
}

setup_train() {
  echo "========== 训练环境: ${TRAIN_ENV} =========="
  ensure_env "$TRAIN_ENV" 3.10

  echo "[INFO] 安装 requirements.txt (PyTorch cu128)..."
  echo "[NOTE] lmdeploy>=0.12 才兼容 torch 2.9（0.11.x 要求 torch<=2.8）"
  conda run -n "$TRAIN_ENV" pip install --upgrade pip
  pip_in "$TRAIN_ENV" install -r "${ROOT}/requirements.txt" \
    --extra-index-url "${PYTORCH_INDEX}"

  install_flash_attn "$TRAIN_ENV"

  if [[ "$SETUP_SGLANG" -eq 1 ]]; then
    setup_sglang "$TRAIN_ENV"
  fi
}

setup_eval() {
  echo "========== 评测环境: ${EVAL_ENV} =========="
  ensure_env "$EVAL_ENV" 3.10

  echo "[INFO] 安装 opencompass..."
  pip_in "$EVAL_ENV" install -e "${ROOT}/opencompass"

  if [[ ! -d "${HUMAN_EVAL_DIR}/.git" ]]; then
    echo "[INFO] 克隆 human-eval..."
    git clone --depth 1 https://github.com/open-compass/human-eval.git "${HUMAN_EVAL_DIR}"
  fi
  # human-eval setup.py 依赖 pkg_resources；pip 隔离构建环境默认没有它
  pip_in "$EVAL_ENV" install 'setuptools>=65,<81' wheel
  pip_in "$EVAL_ENV" install -e "${HUMAN_EVAL_DIR}" --no-build-isolation

  pip_in "$EVAL_ENV" install math_verify latex2sympy2_extended
}

align_sglang_deps() {
  local env="$1"
  # 与 third_party/sglang v0.5.9 pyproject.toml 对齐，避免 requirements 重装后降级
  echo "[INFO] 对齐 SGLang 依赖 (flashinfer 0.6.3, grpcio>=1.78)..."
  pip_in "$env" install \
    flashinfer-cubin==0.6.3 \
    flashinfer-python==0.6.3 \
    'grpcio>=1.78.0' \
    'grpcio-health-checking>=1.78.0' \
    'grpcio-reflection>=1.78.0'
}

setup_sglang() {
  local env="$1"
  if [[ ! -d "${SGLANG_DIR}/.git" ]]; then
    echo "[INFO] 克隆 SGLang v0.5.9..."
    mkdir -p "$(dirname "${SGLANG_DIR}")"
    git clone --depth 1 -b v0.5.9 https://github.com/sgl-project/sglang.git "${SGLANG_DIR}"
  fi
  echo "[INFO] 在 ${env} 中安装 SGLang..."
  pip_in "$env" install -e "${SGLANG_DIR}/python"
  align_sglang_deps "$env"
}

write_env_file() {
  mkdir -p "$(dirname "${ENV_FILE}")"
  cat > "${ENV_FILE}" <<EOF
# source 此文件以加载 DARE 常用环境变量
# usage: source scripts/dare_env.sh && conda activate ${TRAIN_ENV}

export DARE_ROOT="${ROOT}"
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1
export WANDB_PROJECT="DARE"
export WANDB_MODE="\${WANDB_MODE:-offline}"
export WANDB_RESUME="allow"
export HF_HOME="\${HF_HOME:-\$HOME/.cache/huggingface}"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export COMPASS_DATA_CACHE="\${COMPASS_DATA_CACHE:-opencompass}"
# flash-attn 源码编译 / 部分 CUDA 扩展需要
export CUDA_HOME="\${CONDA_PREFIX}"
export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
cd "\${DARE_ROOT}"
EOF
  echo "[INFO] 已写入环境变量脚本: ${ENV_FILE}"
}

verify() {
  echo "========== 环境校验 =========="
  if [[ "$SETUP_TRAIN" -eq 1 ]]; then
    conda run -n "$TRAIN_ENV" python -c "
import torch, flash_attn
print('[OK] ${TRAIN_ENV}: torch', torch.__version__, 'cuda=', torch.version.cuda, 'gpus=', torch.cuda.device_count())
print('[OK] flash_attn', flash_attn.__version__)
"
    conda run -n "$TRAIN_ENV" bash -c "cd '${ROOT}' && python -c 'import verl; print(\"[OK] verl import\")'"
    if [[ "$SETUP_SGLANG" -eq 1 ]]; then
      conda run -n "$TRAIN_ENV" python -c "import sglang; print('[OK] sglang import')" 2>/dev/null \
        || echo "[WARN] sglang 未安装或导入失败（SDAR/LLaDA2 推理需要）"
    fi
  fi
  if [[ "$SETUP_EVAL" -eq 1 ]]; then
    conda run -n "$EVAL_ENV" python -c "import opencompass; print('[OK] opencompass import')"
  fi
}

main() {
  echo "[INFO] DARE_ROOT=${ROOT}"
  echo "[INFO] CONDA_ROOT=${CONDA_ROOT}"
  nvidia-smi -L 2>/dev/null || echo "[WARN] nvidia-smi 不可用"

  if [[ "$FLASH_ATTN_ONLY" -eq 1 ]]; then
    ensure_env "$TRAIN_ENV" 3.10
    install_flash_attn "$TRAIN_ENV"
  else
    [[ "$SETUP_TRAIN" -eq 1 ]] && setup_train
    [[ "$SETUP_EVAL" -eq 1 ]]  && setup_eval
  fi
  write_env_file
  verify

  cat <<EOF

========== 完成 ==========
训练:  source ${ENV_FILE} && conda activate ${TRAIN_ENV}
评测:  conda activate ${EVAL_ENV} && cd ${ROOT}/opencompass

示例:
  bash recipe/llada/run_d1_llada_8b_instruct.sh --task math
  bash scripts/eval_llada1dot5.sh --task mmlu
EOF
}

main "$@"
