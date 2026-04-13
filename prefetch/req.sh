#!/usr/bin/env bash

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30304}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
MODEL="${MODEL:-/data/dark/tmp/workspace/DeepSeekV3.2}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
PROMPT="${PROMPT:-}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-768}"
DECODE_TOKENS="${DECODE_TOKENS:-${MAX_TOKENS}}"
THINKING="${THINKING:-true}"
SEPARATE_REASONING="${SEPARATE_REASONING:-true}"
SEED="${SEED:-42}"
NUM_RUNS="${NUM_RUNS:-2}"
SHOW_RAW_RESPONSE="${SHOW_RAW_RESPONSE:-0}"
USE_LONG_CONTEXT_DEFAULT="${USE_LONG_CONTEXT_DEFAULT:-1}"
LONG_CONTEXT_BLOCKS="${LONG_CONTEXT_BLOCKS:-96}"
PREFILL_TOKENS="${PREFILL_TOKENS:-}"
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
  DECODE_TOKENS="${DECODE_TOKENS}" \
  THINKING="${THINKING}" \
  SEPARATE_REASONING="${SEPARATE_REASONING}" \
  SEED="${SEED}" \
  USE_LONG_CONTEXT_DEFAULT="${USE_LONG_CONTEXT_DEFAULT}" \
  LONG_CONTEXT_BLOCKS="${LONG_CONTEXT_BLOCKS}" \
  PREFILL_TOKENS="${PREFILL_TOKENS}" \
  TARGET_BLOCK_INDEX="${TARGET_BLOCK_INDEX}" \
  OUTPUT_LINE_COUNT="${OUTPUT_LINE_COUNT}" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

from transformers import AutoTokenizer


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_long_context_prompt(
    num_blocks: int, target_idx: int, output_line_count: int
) -> str:
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


def build_prefill_target_prompt(
    model: str, target_tokens: int, target_idx: int, output_line_count: int
) -> str:
    target_tokens = max(1, target_tokens)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    low = 1
    high = 1
    prompt = build_long_context_prompt(high, target_idx, output_line_count)
    token_count = len(tokenizer.encode(prompt, add_special_tokens=False))

    while token_count < target_tokens:
        low = high + 1
        high *= 2
        prompt = build_long_context_prompt(high, target_idx, output_line_count)
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))

    best_prompt = prompt
    best_distance = abs(token_count - target_tokens)

    left = max(1, low)
    right = high
    while left <= right:
        mid = (left + right) // 2
        prompt = build_long_context_prompt(mid, target_idx, output_line_count)
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        distance = abs(token_count - target_tokens)
        if distance <= best_distance:
            best_prompt = prompt
            best_distance = distance
        if token_count < target_tokens:
            left = mid + 1
        elif token_count > target_tokens:
            right = mid - 1
        else:
            return prompt

    return best_prompt


def infer_model_type(model: str) -> str | None:
    config_path = Path(model) / "config.json"
    if not config_path.exists():
        return None
    with config_path.open(encoding="utf-8") as f:
        return json.load(f).get("model_type")


def render_prompt(messages, model: str, thinking: bool) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    thinking_mode = "thinking" if thinking else "chat"

    if infer_model_type(model) == "deepseek_v32":
        from sglang.srt.entrypoints.openai.encoding_dsv32 import encode_messages

        return encode_messages(messages, thinking_mode=thinking_mode)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_dict=False,
        thinking=thinking,
    )


messages = []
system_prompt = os.environ.get("SYSTEM_PROMPT", "")
if system_prompt:
    messages.append({"role": "system", "content": system_prompt})

prompt = os.environ.get("PROMPT", "")
prefill_tokens = os.environ.get("PREFILL_TOKENS", "").strip()
if not prompt and prefill_tokens:
    prompt = build_prefill_target_prompt(
        os.environ["MODEL"],
        int(prefill_tokens),
        int(os.environ["TARGET_BLOCK_INDEX"]),
        int(os.environ["OUTPUT_LINE_COUNT"]),
    )
elif not prompt and parse_bool(os.environ["USE_LONG_CONTEXT_DEFAULT"]):
    prompt = build_long_context_prompt(
        int(os.environ["LONG_CONTEXT_BLOCKS"]),
        int(os.environ["TARGET_BLOCK_INDEX"]),
        int(os.environ["OUTPUT_LINE_COUNT"]),
    )
elif not prompt:
    prompt = "Please briefly explain the purpose of HiSparse."

messages.append({"role": "user", "content": prompt})
rendered_prompt = render_prompt(
    messages,
    os.environ["MODEL"],
    parse_bool(os.environ["THINKING"]),
)

body = {
    "text": rendered_prompt,
    "sampling_params": {
        "temperature": float(os.environ["TEMPERATURE"]),
        "max_new_tokens": int(os.environ["DECODE_TOKENS"]),
        "sampling_seed": int(os.environ["SEED"]),
    },
}

print(json.dumps(body, ensure_ascii=False))
PY
)"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

for run_idx in $(seq 1 "${NUM_RUNS}"); do
  curl -sS \
    -X POST "${BASE_URL}/generate" \
    -H 'Content-Type: application/json' \
    --data-binary "${payload}" \
    > "${tmp_dir}/response_${run_idx}.json"
done

TMP_DIR="${tmp_dir}" \
NUM_RUNS="${NUM_RUNS}" \
SHOW_RAW_RESPONSE="${SHOW_RAW_RESPONSE}" \
SEPARATE_REASONING="${SEPARATE_REASONING}" \
MODEL="${MODEL}" \
python3 - <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from sglang.srt.parser.reasoning_parser import ReasoningParser


def short_hash(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def preview(text, limit=120):
    text = (text or "").replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


def extract_response(resp):
    if "text" not in resp:
        return {"raw": resp}
    text = resp.get("text") or ""
    reasoning = ""
    content = text
    model_is_deepseek_v32 = "deepseekv3.2" in os.environ["MODEL"].lower() or "deepseek_v32" in os.environ["MODEL"].lower()
    separate_reasoning = os.environ["SEPARATE_REASONING"].strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if separate_reasoning and model_is_deepseek_v32:
        reasoning, content = ReasoningParser("deepseek-v3").parse_non_stream(text)
    return {
        "text": text,
        "content": content or "",
        "reasoning_content": reasoning or "",
        "finish_reason": (resp.get("meta_info") or {}).get("finish_reason"),
        "usage": resp.get("meta_info") or {},
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
