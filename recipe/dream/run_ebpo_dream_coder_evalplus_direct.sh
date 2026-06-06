#!/bin/bash
# Dream-Coder EBPO on EvalPlus HumanEval+MBPP — wrapper over the BGPO direct script.
# EBPO differs only in forward_process (_forward_process_ebpo); rollout/reward/actor match BGPO.
#
# Usage:
#   bash recipe/dream/run_ebpo_dream_coder_evalplus_direct.sh --smoke --engine sglang
#   bash recipe/dream/run_ebpo_dream_coder_evalplus_direct.sh --engine sglang
set -euo pipefail

_script_path="${BASH_SOURCE[0]}"
[[ "${_script_path}" != /* ]] && _script_path="$(pwd)/${_script_path}"
SCRIPT_DIR="$(cd "$(dirname "${_script_path}")" && pwd)"
unset _script_path

exec bash "${SCRIPT_DIR}/run_bgpo_dream_coder_evalplus_direct.sh" "$@" --algorithm ebpo
