import argparse
import atexit
import json
import os
import shutil
import sys
import time
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SYSTEM_PROMPT = os.path.join(HERE, "system_prompt.md")

STATIC_SAMPLES = {
    "hello": ("Hello! Who are you?", None),
    "aime25": (
        "Find the sum of all integer bases b > 9 for which 17_b is a divisor "
        "of 97_b.",
        "70  (AIME 2025 I Problem 1)",
    ),
}
SAMPLE_CHOICES = list(STATIC_SAMPLES) + ["longbench-v2"]


def load_longbench_v2(
    index: int, difficulty: str | None, length: str | None
) -> tuple[str, str]:
    """Fetch one THUDM/LongBench-v2 example and render it in the official
    zero-shot CoT prompt format.
    `difficulty` may be None, "easy", or "hard".
    `length` may be None, "short", "medium", or "long"."""
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench-v2", split="train")
    if difficulty:
        ds = ds.filter(lambda r: r["difficulty"] == difficulty)
    if length:
        ds = ds.filter(lambda r: r["length"] == length)
    if index >= len(ds):
        raise IndexError(
            f"longbench-v2 index {index} out of range (filtered size={len(ds)})"
        )
    rec = ds[index]
    print(
        f"[longbench-v2] id={rec['_id']} domain={rec['domain']} "
        f"difficulty={rec['difficulty']} length={rec['length']} "
        f"answer={rec['answer']} context_chars={len(rec['context'])}",
        flush=True,
    )
    prompt_text = (
        f"{rec['context']}\n\n"
        f"What is the correct answer to this question: {rec['question']}\n"
        f"Choices:\n"
        f"(A) {rec['choice_A']}\n"
        f"(B) {rec['choice_B']}\n"
        f"(C) {rec['choice_C']}\n"
        f"(D) {rec['choice_D']}\n\n"
        f"Let's think step by step, and the correct answer is:"
    )
    gold_letter = rec["answer"]
    gold_text = rec.get(f"choice_{gold_letter}", "")
    gold_label = (
        f"({gold_letter}) {gold_text}  "
        f"[id={rec['_id']} domain={rec['domain']} "
        f"difficulty={rec['difficulty']} length={rec['length']}]"
    )
    return prompt_text, gold_label


parser = argparse.ArgumentParser()
parser.add_argument("--sample", choices=SAMPLE_CHOICES, default="hello")
parser.add_argument("--url", default="http://127.0.0.1:35059/v1/chat/completions")
parser.add_argument("--max-tokens", type=int, default=32768)
parser.add_argument(
    "--system-prompt-file",
    default=DEFAULT_SYSTEM_PROMPT,
    help="Path to a text file used as the system prompt. Pass '' to disable.",
)
parser.add_argument(
    "--longbench-index",
    type=int,
    default=0,
    help="Index into the (optionally filtered) LongBench-v2 train split.",
)
parser.add_argument(
    "--longbench-difficulty",
    choices=["easy", "hard"],
    default=None,
    help="Filter LongBench-v2 by difficulty. Default: no filter.",
)
parser.add_argument(
    "--longbench-length",
    choices=["short", "medium", "long"],
    default="short",
    help=(
        "Filter LongBench-v2 by length bucket. Default: 'short' "
        "(~<32k tokens) so records fit in the default DeepSeek context."
    ),
)
args = parser.parse_args()

if args.sample == "longbench-v2":
    prompt, expected_answer = load_longbench_v2(
        args.longbench_index, args.longbench_difficulty, args.longbench_length
    )
else:
    prompt, expected_answer = STATIC_SAMPLES[args.sample]

system_prompt = ""
if args.system_prompt_file:
    with open(args.system_prompt_file) as f:
        system_prompt = f.read()
    print(
        f"=== system prompt ({args.system_prompt_file}, "
        f"{len(system_prompt)} chars) ===\n"
        f"{system_prompt[:200]}...\n[truncated in display]\n",
        flush=True,
    )

if len(prompt) > 2000:
    print(
        f"=== prompt ({args.sample}, {len(prompt)} chars) ===\n"
        f"{prompt[:800]}\n...[truncated {len(prompt) - 1200} chars]...\n"
        f"{prompt[-400:]}\n",
        flush=True,
    )
else:
    print(f"=== prompt ({args.sample}) ===\n{prompt}\n", flush=True)

messages = []
if system_prompt:
    messages.append({"role": "system", "content": system_prompt})
messages.append({"role": "user", "content": prompt})


# --- bottom-row status line (only when stdout is a tty) -------------------
IS_TTY = sys.stdout.isatty()
cols, rows = shutil.get_terminal_size(fallback=(120, 40))

if IS_TTY:
    # reserve the bottom row: scroll region = rows 1..rows-1
    sys.stdout.write(f"\033[1;{rows - 1}r")
    # move cursor into the scroll region so new output doesn't land on status row
    sys.stdout.write(f"\033[{rows - 1};1H")
    sys.stdout.flush()

    def _restore():
        sys.stdout.write("\033[r")  # reset scroll region
        sys.stdout.write(f"\033[{rows};1H\033[K")  # clear status row
        sys.stdout.flush()

    atexit.register(_restore)

def set_status(text):
    if not IS_TTY:
        return
    # save cursor → goto bottom row → clear → write → restore cursor
    sys.stdout.write(f"\0337\033[{rows};1H\033[K{text[:cols - 1]}\0338")
    sys.stdout.flush()


# --- stream ---------------------------------------------------------------
t_request = time.perf_counter()
resp = requests.post(
    args.url,
    json={
        "model": "/data/dark/workspace/DeepSeekV3.2",
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": True},
    },
    stream=True,
    timeout=None,
)
if not resp.ok:
    # server returns JSON error body for 4xx/5xx — print it before raising
    print(f"\n=== HTTP {resp.status_code} error body ===", file=sys.stderr)
    try:
        print(resp.text, file=sys.stderr, flush=True)
    finally:
        resp.raise_for_status()

section = None
usage = None
decode_count = 0
t_first = None
t_last = None

def switch(new):
    global section
    if section != new:
        if section is not None:
            print()
        print(f"=== {new} ===", flush=True)
        section = new

set_status("prefill=? decode≈0 tps=?")

for raw in resp.iter_lines(decode_unicode=True):
    if not raw or not raw.startswith("data:"):
        continue
    payload = raw[len("data:"):].strip()
    if payload == "[DONE]":
        break
    chunk = json.loads(payload)
    if chunk.get("usage"):
        usage = chunk["usage"]
    for choice in chunk.get("choices", []):
        delta = choice.get("delta") or {}
        piece = delta.get("reasoning_content") or delta.get("content")
        if not piece:
            continue
        if delta.get("reasoning_content"):
            switch("reasoning")
        else:
            switch("answer")
        sys.stdout.write(piece)
        sys.stdout.flush()
        decode_count += 1
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        t_last = now
    pt = usage.get("prompt_tokens") if usage else "?"
    ct = usage.get("completion_tokens") if usage else decode_count
    tt = usage.get("total_tokens") if usage else "?"
    # live decode TPS = (tokens after first) / (elapsed since first)
    if t_first is not None and t_last is not None and decode_count > 1:
        tps_live = (decode_count - 1) / max(t_last - t_first, 1e-6)
        tps_str = f"{tps_live:.1f} tok/s"
    else:
        tps_str = "?"
    set_status(
        f"prefill={pt}  decode={ct}{' (approx)' if not usage else ''}  "
        f"total={tt}  tps={tps_str}"
    )

t_end = time.perf_counter()

# final — usage is authoritative from the last pre-[DONE] chunk
print("\n\n=== usage ===")
if usage:
    prompt_toks = usage.get("prompt_tokens")
    completion_toks = usage.get("completion_tokens")
    total_toks = usage.get("total_tokens")
    print(f"prefill (prompt) tokens: {prompt_toks}")
    print(f"decode (completion) tokens: {completion_toks}")
    print(f"total tokens (context len): {total_toks}")
else:
    completion_toks = decode_count
    print("(no usage reported — did stream_options.include_usage get stripped?)")

print("\n=== timing ===")
if t_first is not None:
    ttft = t_first - t_request
    decode_dur = (t_last - t_first) if t_last and t_last > t_first else 0.0
    print(f"TTFT (request → first token): {ttft * 1000:.1f} ms")
    print(f"decode duration:              {decode_dur:.3f} s ({decode_count} streamed chunks)")
    if completion_toks and decode_dur > 0:
        # TPS excludes the first token (its latency is TTFT, not decode)
        tps = (completion_toks - 1) / decode_dur
        print(f"decode TPS (completion-1 / decode_dur): {tps:.2f} tok/s")
    total_dur = t_end - t_request
    print(f"total wall time:              {total_dur:.3f} s")
else:
    print("(no tokens streamed)")

if expected_answer is not None:
    print(f"\n=== expected answer ===\n{expected_answer}")
