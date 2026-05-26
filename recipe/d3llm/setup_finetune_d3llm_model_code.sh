#!/bin/bash
# Copy Dream HF modeling code into models/finetune_d3LLM for offline trust_remote_code.
set -euo pipefail

DARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D3LLM_ROOT="${D3LLM_ROOT:-/home/u-liujc/Codes/d3LLM}"
MODEL_DIR="${MODEL_DIR:-${DARE_ROOT}/models/finetune_d3LLM}"
SRC="${D3LLM_ROOT}/utils/utils_Dream/model"

if [[ ! -d "${SRC}" ]]; then
  echo "[ERROR] Missing ${SRC}. Set D3LLM_ROOT to your d3LLM clone."
  exit 1
fi

mkdir -p "${MODEL_DIR}"
cp -f "${SRC}/configuration_dream.py" "${SRC}/modeling_dream.py" "${SRC}/generation_utils.py" "${MODEL_DIR}/"

# Ensure config.json points at local modules (idempotent).
python3 - <<'PY' "${MODEL_DIR}/config.json"
import json, sys
path = sys.argv[1]
with open(path) as f:
    cfg = json.load(f)
cfg["auto_map"] = {
    "AutoConfig": "configuration_dream.DreamConfig",
    "AutoModel": "modeling_dream.DreamModel",
}
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY

echo "[OK] Dream modeling files installed under ${MODEL_DIR}"
ls -la "${MODEL_DIR}"/*.py
