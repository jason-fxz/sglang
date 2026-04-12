#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export PYTHONPATH="${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/home/ubuntu/workspace/models/DeepSeek-V3.2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP="${TP:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"
ENABLE_TRUNCATED_LAYERS="${ENABLE_TRUNCATED_LAYERS:-1}"
TRUNCATED_NUM_HIDDEN_LAYERS="${TRUNCATED_NUM_HIDDEN_LAYERS:-4}"

# ── HiSparse config (built from individual knobs) ────────────────────
HISPARSE_TOP_K="${HISPARSE_TOP_K:-2048}"
HISPARSE_DEVICE_BUFFER_SIZE="${HISPARSE_DEVICE_BUFFER_SIZE:-2048}"
HISPARSE_HOST_TO_DEVICE_RATIO="${HISPARSE_HOST_TO_DEVICE_RATIO:-2}"
# Stage B prefetch knobs (set ENABLE_PREFETCH=1 to turn on)
ENABLE_PREFETCH="${ENABLE_PREFETCH:-0}"
PREFETCH_TOPK="${PREFETCH_TOPK:-${HISPARSE_TOP_K}}"
NUM_MAX_PREFETCH="${NUM_MAX_PREFETCH:-512}"

_hisparse_json="{\"top_k\":${HISPARSE_TOP_K},\"device_buffer_size\":${HISPARSE_DEVICE_BUFFER_SIZE},\"host_to_device_ratio\":${HISPARSE_HOST_TO_DEVICE_RATIO}"
if [[ "${ENABLE_PREFETCH}" == "1" ]]; then
  _hisparse_json="${_hisparse_json},\"enable_prefetch\":true,\"prefetch_topk\":${PREFETCH_TOPK},\"num_max_prefetch\":${NUM_MAX_PREFETCH}"
fi
_hisparse_json="${_hisparse_json}}"
HISPARSE_CONFIG="${HISPARSE_CONFIG:-${_hisparse_json}}"

has_manual_model_override=0
for arg in "$@"; do
  if [[ "${arg}" == "--json-model-override-args" || "${arg}" == --json-model-override-args=* ]]; then
    has_manual_model_override=1
    break
  fi
done

launch_args=(
  -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --tp "${TP}"
  --trust-remote-code
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --disable-radix-cache
  --enable-hisparse
  --hisparse-config "${HISPARSE_CONFIG}"
  --reasoning-parser deepseek-v3
  --tool-call-parser deepseekv32
)

if [[ "${ENABLE_TRUNCATED_LAYERS}" == "1" && "${has_manual_model_override}" == "0" ]]; then
  truncated_model_override="$(printf '{"num_hidden_layers": %s}' "${TRUNCATED_NUM_HIDDEN_LAYERS}")"
  launch_args+=(
    --json-model-override-args "${truncated_model_override}"
  )
fi

launch_args+=("$@")

exec "${PYTHON_BIN}" "${launch_args[@]}"
