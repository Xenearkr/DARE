#!/bin/bash
set -e

export TORCHDYNAMO_DISABLE=1
export HF_HOME=
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HUB_OFFLINE=1
export COMPASS_DATA_CACHE=opencompass
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

model=${model:-SDAR-4B-Chat}
engine=${engine:-hf}

if [ -z "${task}" ]; then
  echo "Usage: bash eval.sh ${task}"
  echo "Optional task: mmlu, mmlupro, hellaswag, arcc, gsm8k_confidence math_confidence gpqa_confidence humaneval_logits mbpp_confidence gsm8k_short math_short"
  exit 1
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
exp_name="eval_${model}_${task}"
log_dir=./logs/EVAL/${exp_name}
mkdir -p ${log_dir}

if [ "${engine}" = "lmdeploy" ]; then
  prefix="lmdeploy_"
elif [ "${engine}" = "sglang" ]; then
  prefix="sglang_"
else
  prefix=""
fi

# task Execution Map
case "${task}" in
  mmlu)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_mmlu_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_mmlu_length4096
    ;;
  mmlupro)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_mmlupro_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_mmlupro_length4096
    ;;
  hellaswag)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_hellaswag_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_hellaswag_length4096
    ;;
  arcc)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_arcc_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_arcc_length4096
    ;;
  gpqa)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_gpqa_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_gpqa_length4096
    ;;
  humaneval)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_humaneval_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_humaneval_length4096
    ;;
  mbpp)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_mbpp_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_mbpp_length4096
    ;;
  gsm8k)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_gsm8k_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_gsm8k_length4096
    ;;
  math)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_math_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_math_length4096
    ;;
  olympiadbench)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_olympiadbench_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_olympiadbench_length4096
    ;;
  aime2024)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_aime2024_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_aime2024_length4096
    ;;
  aime2025)
    py_script=sdar_examples/${prefix}sdar_4b_chat_gen_aime2025_length4096.py
    work_dir=outputs/${prefix}sdar_4b_chat_gen_aime2025_length4096
    ;;
  *)
    echo "Unknown task: ${task}"
    exit 1
    ;;
esac

echo "task: ${task}"
echo "model: ${model}"
echo "Script: ${py_script}"
echo "Work Dir: ${work_dir}"
echo "Log Dir: ${log_dir}"

python run.py "${py_script}" -w "${work_dir}" \
>> "${log_dir}/eval-${task}-${timestamp}.out" \
2>> "${log_dir}/eval-${task}-${timestamp}.err" &