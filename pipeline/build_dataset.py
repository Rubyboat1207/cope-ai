"""
Builds sliding-window next-message-prediction training examples from
messages.jsonl. Every message in the timeline (after the first) becomes one
training example: the target is that message (who sent it, how long after
the previous message, and what it said), and the input is as much of the
preceding timeline as fits in the token budget, snapped to whole messages
(never split mid-message).
"""
import json
from pathlib import Path
from datetime import datetime

from transformers import AutoTokenizer

INPUT_FILE = Path("messages.jsonl")
OUTPUT_FILE = Path("finetune_dataset.jsonl")
MODEL_PATH = "models/Qwen2.5-3B-Instruct"

MAX_CONTEXT_TOKENS = 1024
MAX_TARGET_TOKENS = 128  # reserved out of the budget for the message being predicted
TARGET_MARKER = "\n<|next|>\n"


def load_messages():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r["id"])
    return records


def format_gap(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"+{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"+{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"+{hours}h"
    days = hours // 24
    return f"+{days}d"


def message_text(record: dict) -> str:
    text = (record["content"] or "").strip()
    if record["attachments"]:
        urls = " ".join(a["url"] for a in record["attachments"])
        text = f"{text} [attachment: {urls}]".strip()
    return text


def format_line(record: dict, gap_seconds: float) -> str:
    return f"[{format_gap(gap_seconds)}] {record['author_name']}: {message_text(record)}"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    messages = load_messages()

    lines = []
    gaps = []
    prev_ts = None
    for m in messages:
        ts = datetime.fromisoformat(m["timestamp"])
        gap = 0.0 if prev_ts is None else (ts - prev_ts).total_seconds()
        gaps.append(gap)
        lines.append(format_line(m, gap))
        prev_ts = ts

    token_lens = [len(t) for t in tokenizer([l + "\n" for l in lines])["input_ids"]]

    written = 0
    skipped_no_context = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for i in range(1, len(lines)):
            target_line = lines[i]
            target_len = token_lens[i]
            if target_len > MAX_TARGET_TOKENS:
                # message itself is too long to reasonably predict in full; skip as a target
                continue

            budget = MAX_CONTEXT_TOKENS - target_len
            context_lines = []
            used = 0
            j = i - 1
            while j >= 0 and used + token_lens[j] <= budget:
                context_lines.append(lines[j])
                used += token_lens[j]
                j -= 1
            context_lines.reverse()

            if not context_lines:
                skipped_no_context += 1
                continue

            prompt = "\n".join(context_lines) + TARGET_MARKER
            record = {"prompt": prompt, "completion": target_line}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} examples to {OUTPUT_FILE}")
    print(f"Skipped {skipped_no_context} messages with no room for any context")


if __name__ == "__main__":
    main()
