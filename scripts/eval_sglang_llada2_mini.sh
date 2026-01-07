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

model=${model:-LLaDA2.0-Mini}
engine=${engine:-hf}

if [ -z "${task}" ]; then
  echo "Usage: bash eval.sh ${task}"
  echo "Optional task: mmlu, mmlupro, hellaswag, arcc, gsm8k math gpqa humaneval mbpp olympiad aime2024 aime2025"
  exit 1
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
exp_name="eval_${model}_${task}"
log_dir=logs/EVAL/${exp_name}
mkdir -p ${log_dir}

# task Execution Map
case "${task}" in
  mmlu)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_mmlu.py
    work_dir=outputs/sglang_llada2_mini_mmlu
    ;;
  mmlupro)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_mmlupro.py
    work_dir=outputs/sglang_llada2_mini_mmlupro
    ;;
  hellaswag)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_hellaswag.py
    work_dir=outputs/sglang_llada2_mini_hellaswag
    ;;
  gpqa)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_gpqa.py
    work_dir=outputs/sglang_llada2_mini_gpqa
    ;;
  arcc)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_arcc.py
    work_dir=outputs/sglang_llada2_mini_arcc
    ;;
  mbpp)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_mbpp.py
    work_dir=outputs/sglang_llada2_mini_mbpp
    ;;
  humaneval)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_humaneval.py
    work_dir=outputs/sglang_llada2_mini_gen_humaneval
    ;;
  gsm8k)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_gsm8k.py
    work_dir=outputs/sglang_llada2_mini_gen_gsm8k
    ;;
  math)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_math.py
    work_dir=outputs/sglang_llada2_mini_gen_math
    ;;
  olympiad)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_olympiadbench.py
    work_dir=outputs/sglang_llada2_mini_gen_olympiadbench
    ;;
  aime2024)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_aime2024.py
    work_dir=outputs/sglang_llada2_mini_gen_aime2024
    ;;
  aime2025)
    py_script=sglang_llada2_mini_examples/sglang_llada2_mini_gen_aime2025.py
    work_dir=outputs/sglang_llada2_mini_gen_aime2025
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

python  run.py "${py_script}" -w "${work_dir}" \
>> "${log_dir}/eval-${task}-${timestamp}.out" \
2>> "${log_dir}/eval-${task}-${timestamp}.err" &
