#!/bin/bash
# Phase 3B: run HF and SGLang verify shards on 4 GPUs in parallel, then merge.
set -euo pipefail

DARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${DARE_ROOT}/logs/DARE/verify_3b_$(date +%Y%m%d_%H%M%S)}"
NUM_TASKS="${NUM_TASKS:-8}"
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "${OUT_DIR}"
echo "[INFO] output_dir=${OUT_DIR} num_tasks=${NUM_TASKS} num_shards=${NUM_SHARDS}"

unset PYTORCH_CUDA_ALLOC_CONF

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="${i}" python3 "${DARE_ROOT}/recipe/d3llm/verify_sglang_dream_rollout.py" \
    --backend hf \
    --shard-id "${i}" \
    --num-shards "${NUM_SHARDS}" \
    --num-tasks "${NUM_TASKS}" \
    --temperature 0.0 \
    --output-json "${OUT_DIR}/hf_shard${i}.json" \
    > "${OUT_DIR}/hf_shard${i}.log" 2>&1 &
  PIDS+=($!)
  echo "[INFO] HF shard ${i} on GPU ${i} pid=$!"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=1
done
if [[ "${FAIL}" -ne 0 ]]; then
  echo "[ERROR] HF shard failed; see ${OUT_DIR}/hf_shard*.log"
  exit 1
fi
echo "[INFO] HF shards done"

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES="${i}" python3 "${DARE_ROOT}/recipe/d3llm/verify_sglang_dream_rollout.py" \
    --backend sglang \
    --shard-id "${i}" \
    --num-shards "${NUM_SHARDS}" \
    --num-tasks "${NUM_TASKS}" \
    --temperature 0.0 \
    --mem-fraction-static 0.45 \
    --output-json "${OUT_DIR}/sglang_shard${i}.json" \
    > "${OUT_DIR}/sglang_shard${i}.log" 2>&1 &
  PIDS+=($!)
  echo "[INFO] SGLang shard ${i} on GPU ${i} pid=$!"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=1
done
if [[ "${FAIL}" -ne 0 ]]; then
  echo "[ERROR] SGLang shard failed; see ${OUT_DIR}/sglang_shard*.log"
  exit 1
fi
echo "[INFO] SGLang shards done"

python3 "${DARE_ROOT}/recipe/d3llm/verify_sglang_dream_rollout.py" \
  --merge-json "${OUT_DIR}/hf_shard*.json" "${OUT_DIR}/sglang_shard*.json" \
  | tee "${OUT_DIR}/merge_summary.log"

echo "[INFO] Phase 3B complete. Artifacts: ${OUT_DIR}"
