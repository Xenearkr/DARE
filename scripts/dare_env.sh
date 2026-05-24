# source 此文件以加载 DARE 常用环境变量
# usage: source scripts/dare_env.sh && conda activate DARE

export DARE_ROOT="/home/u-liujc/Codes/DARE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1
export WANDB_PROJECT="DARE"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_RESUME="allow"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export COMPASS_DATA_CACHE="${COMPASS_DATA_CACHE:-opencompass}"
# flash-attn 源码编译 / 部分 CUDA 扩展需要
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
cd "${DARE_ROOT}"
