#!/bin/bash
set -e

export TORCHDYNAMO_DISABLE=1
export HF_HOME=
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_OFFLINE=1
export COMPASS_DATA_CACHE=
cd opencompass

# parameter parsing
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --task)
      task="$2"
      shift; shift
      ;;
    --model)
      model="$2"
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

model=${model:-SDAR-30B-A3B-Chat}
engine=${engine:-hf}

if [ -z "${task}" ]; then
  echo "Usage: bash eval_sdar_30b_a3b_chat.sh --task <task> [--engine hf|sglang|lmdeploy] [--model <model_name>]"
  echo "Supported tasks: mmlu, mmlupro, hellaswag, arcc, gpqa, humaneval, mbpp, gsm8k, math, olympiad, aime2024, aime2025"
  exit 1
fi

if [ "${engine}" = "lmdeploy" ]; then
  prefix="lmdeploy_"
elif [ "${engine}" = "sglang" ]; then
  prefix="sglang_"
else
  prefix=""
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
exp_name="eval_${model}_${task}"
log_dir=./logs/EVAL/${exp_name}
mkdir -p "${log_dir}"

case "${task}" in
  mmlu)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_mmlu.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_mmlu
    ;;
  mmlupro)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_mmlupro.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_mmlupro
    ;;
  hellaswag)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_hellaswag.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_hellaswag
    ;;
  arcc)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_arcc.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_arcc
    ;;
  gpqa)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_gpqa.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_gpqa
    ;;
  humaneval)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_humaneval.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_humaneval
    ;;
  mbpp)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_mbpp.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_mbpp
    ;;
  gsm8k)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_gsm8k.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_gsm8k
    ;;
  math)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_math.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_math
    ;;
  olympiad|olympiadbench)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_olympiadbench.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_olympiadbench
    ;;
  aime2024)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_aime2024.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_aime2024
    ;;
  aime2025)
    py_script=sdar_30b_a3b_examples/${prefix}sdar_30b_a3b_chat_gen_aime2025.py
    work_dir=outputs/${prefix}sdar_30b_a3b_chat_gen_aime2025
    ;;
  *)
    echo "Unknown task: ${task}"
    exit 1
    ;;
esac

echo "task: ${task}"
echo "model: ${model}"
echo "engine: ${engine}"
echo "Script: ${py_script}"
echo "Work Dir: ${work_dir}"
echo "Log Dir: ${log_dir}"

python run.py "${py_script}" -w "${work_dir}" \
>> "${log_dir}/eval-${task}-${timestamp}.out" \
2>> "${log_dir}/eval-${task}-${timestamp}.err"
