#!/usr/bin/env bash

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
MODEL="${MODEL:-/home/ubuntu/workspace/models/DeepSeek-V3.2}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
PROMPT="${PROMPT:-}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-768}"
THINKING="${THINKING:-true}"
SEPARATE_REASONING="${SEPARATE_REASONING:-true}"
SEED="${SEED:-42}"
NUM_RUNS="${NUM_RUNS:-2}"
SHOW_RAW_RESPONSE="${SHOW_RAW_RESPONSE:-0}"
USE_LONG_CONTEXT_DEFAULT="${USE_LONG_CONTEXT_DEFAULT:-1}"
LONG_CONTEXT_BLOCKS="${LONG_CONTEXT_BLOCKS:-96}"
TARGET_BLOCK_INDEX="${TARGET_BLOCK_INDEX:-63}"
OUTPUT_LINE_COUNT="${OUTPUT_LINE_COUNT:-96}"

if [[ "$#" -gt 0 ]]; then
  PROMPT="$*"
fi

payload="$(
  HOST="${HOST}" \
  PORT="${PORT}" \
  BASE_URL="${BASE_URL}" \
  MODEL="${MODEL}" \
  SYSTEM_PROMPT="${SYSTEM_PROMPT}" \
  PROMPT="${PROMPT}" \
  TEMPERATURE="${TEMPERATURE}" \
  MAX_TOKENS="${MAX_TOKENS}" \
  THINKING="${THINKING}" \
  SEPARATE_REASONING="${SEPARATE_REASONING}" \
  SEED="${SEED}" \
  USE_LONG_CONTEXT_DEFAULT="${USE_LONG_CONTEXT_DEFAULT}" \
  LONG_CONTEXT_BLOCKS="${LONG_CONTEXT_BLOCKS}" \
  TARGET_BLOCK_INDEX="${TARGET_BLOCK_INDEX}" \
  OUTPUT_LINE_COUNT="${OUTPUT_LINE_COUNT}" \
  python3 - <<'PY'
import json
import os


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_long_context_prompt(num_blocks: int, target_idx: int, output_line_count: int) -> str:
    target_idx = max(0, min(target_idx, num_blocks - 1))
    lines = [
        "Below is a long retrieval context used to exercise HiSparse during decode.",
        "Read the whole context carefully.",
    ]
    for i in range(num_blocks):
        checksum_a = (i * 17 + 11) % 10007
        checksum_b = (i * 29 + 7) % 10007
        checksum_c = (i * 43 + 19) % 10007
        line = (
            f"Record {i:04d}: archive segment {i:04d}; "
            f"verification tuple [{checksum_a}, {checksum_b}, {checksum_c}]; "
            f"label token-{i:04d}-delta; "
            f"memo this line exists for long-context sparse-attention retrieval testing."
        )
        if i == target_idx:
            line += f" Special answer token: HISPARSE_TARGET_{target_idx:03d}_ALPHA."
        lines.append(line)
    lines.append(
        "Task: find the special answer token from the target record. "
        f"Then output exactly {output_line_count} lines. "
        "Each line must be formatted as `line NNN: TOKEN`, where TOKEN is the special answer token. "
        "After those lines, output one final line formatted as `target index: IDX`."
    )
    return "\n".join(lines)


messages = []
system_prompt = os.environ.get("SYSTEM_PROMPT", "")
if system_prompt:
    messages.append({"role": "system", "content": system_prompt})

prompt = os.environ.get("PROMPT", "")
if not prompt and parse_bool(os.environ["USE_LONG_CONTEXT_DEFAULT"]):
    prompt = build_long_context_prompt(
        int(os.environ["LONG_CONTEXT_BLOCKS"]),
        int(os.environ["TARGET_BLOCK_INDEX"]),
        int(os.environ["OUTPUT_LINE_COUNT"]),
    )
elif not prompt:
    prompt = "请简单介绍一下 HiSparse 的作用。"

messages.append({"role": "user", "content": prompt})

body = {
    "model": os.environ["MODEL"],
    "messages": messages,
    "temperature": float(os.environ["TEMPERATURE"]),
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "seed": int(os.environ["SEED"]),
    "chat_template_kwargs": {
        "thinking": parse_bool(os.environ["THINKING"]),
    },
    "separate_reasoning": parse_bool(os.environ["SEPARATE_REASONING"]),
}

print(json.dumps(body, ensure_ascii=False))
PY
)"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

for run_idx in $(seq 1 "${NUM_RUNS}"); do
  curl -sS \
    -X POST "${BASE_URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "${payload}" \
    > "${tmp_dir}/response_${run_idx}.json"
done

TMP_DIR="${tmp_dir}" \
NUM_RUNS="${NUM_RUNS}" \
SHOW_RAW_RESPONSE="${SHOW_RAW_RESPONSE}" \
python3 - <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def short_hash(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def preview(text, limit=120):
    text = (text or "").replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


def extract_response(resp):
    if "choices" not in resp or not resp["choices"]:
        return {"raw": resp}
    choice = resp["choices"][0]
    message = choice.get("message", {})
    return {
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "usage": resp.get("usage"),
    }


tmp_dir = Path(os.environ["TMP_DIR"])
num_runs = int(os.environ["NUM_RUNS"])
show_raw = os.environ["SHOW_RAW_RESPONSE"].strip().lower() in {"1", "true", "yes", "on"}

responses = []
for idx in range(1, num_runs + 1):
    path = tmp_dir / f"response_{idx}.json"
    with path.open() as f:
        responses.append(json.load(f))

extracted = [extract_response(resp) for resp in responses]

for idx, item in enumerate(extracted, start=1):
    print(f"=== Run {idx} ===")
    if "raw" in item:
        print(json.dumps(item["raw"], ensure_ascii=False, indent=2))
        continue

    content = item["content"] or ""
    reasoning = item["reasoning_content"] or ""
    usage = item.get("usage") or {}
    prefill_tokens = usage.get("prompt_tokens", "?")
    decode_tokens = usage.get("completion_tokens", "?")
    print(
        f"finish_reason={item['finish_reason']} "
        f"prefill_tokens={prefill_tokens} decode_tokens={decode_tokens}"
    )
    if content:
        print(f"content_preview={preview(content)}")
    if reasoning:
        print(f"reasoning_preview={preview(reasoning)}")
    if show_raw:
        print(json.dumps(responses[idx - 1], ensure_ascii=False, indent=2))

def usage_key(item):
    usage = item.get("usage") or {}
    return (item.get("finish_reason"), usage.get("prompt_tokens"), usage.get("completion_tokens"))

consistent = True
if extracted:
    base = usage_key(extracted[0])
    for item in extracted[1:]:
        if usage_key(item) != base:
            consistent = False
            break

print("=== Compare ===")
print(f"consistent={str(consistent).lower()}")

if not consistent:
    sys.exit(1)
PY
