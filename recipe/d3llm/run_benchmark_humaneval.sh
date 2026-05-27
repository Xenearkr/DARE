#!/bin/bash
# Full HumanEval decode-path benchmark on 4 GPUs (DARE conda env).
set -euo pipefail

DARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${DARE_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE=offline

PYTHON="${PYTHON:-/home/u-liujc/anaconda3/envs/DARE/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=python3
fi

OUT="${1:-logs/benchmarks/humaneval_decode_paths.json}"
mkdir -p "$(dirname "${OUT}")"

echo "[INFO] PYTHON=${PYTHON} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] output=${OUT}"

"${PYTHON}" recipe/d3llm/benchmark_humaneval_decode_paths.py \
  --limit 164 \
  --world-size 4 \
  --paths hf_multiblock,sglang_train,sglang_val \
  --mem-fraction-static 0.45 \
  --output-json "${OUT}" \
  2>&1 | tee "${OUT%.json}.log"
