#!/usr/bin/env bash
# Thin wrapper around launch_hisparse_ds_v32.sh — matches the port used by
# send_request.py (35059), enables Stage B prefetch toggle, and tees both
# stdout and stderr to a timestamped .log file in this directory.
#
# Modes:
#   ENABLE_HISPARSE=1 ENABLE_PREFETCH=0  (default)  — HiSparse on, prefetch off
#   ENABLE_HISPARSE=1 ENABLE_PREFETCH=1             — HiSparse + Stage B prefetch
#   ENABLE_HISPARSE=0                               — baseline, HiSparse off

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/launch_$(date +%Y%m%d_%H%M%S).log}"

echo "[launch.sh] logging to ${LOG_FILE}" >&2

PORT="${PORT:-35059}" \
ENABLE_HISPARSE="${ENABLE_HISPARSE:-1}" \
ENABLE_PREFETCH="${ENABLE_PREFETCH:-0}" \
PYTHONUNBUFFERED=1 \
  stdbuf -oL -eL "${SCRIPT_DIR}/launch_hisparse_ds_v32.sh" "$@" 2>&1 \
  | stdbuf -oL tee "${LOG_FILE}"
