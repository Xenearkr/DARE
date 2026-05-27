#!/usr/bin/env bash
# Re-run aligned sglang_val on 82 HumanEval rows (HF shards must exist).
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-$HOME/anaconda3/envs/DARE/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
"$PYTHON" recipe/d3llm/benchmark_humaneval_pipelines.py \
  --only-sglang-val \
  --inprocess \
  --limit 82 \
  --ngpus 1 \
  --mem-fraction-static 0.55 \
  --hf-shards-dir logs/benchmarks/humaneval_pipelines_latest_shards \
  --output-json logs/benchmarks/humaneval_pipelines_val_aligned.json \
  2>&1 | tee logs/benchmarks/humaneval_pipelines_val_aligned.log
