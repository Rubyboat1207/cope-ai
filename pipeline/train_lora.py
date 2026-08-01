"""
LoRA finetuning of Qwen2.5-3B-Instruct on the sliding-window next-message
dataset, running on a local AMD GPU via ROCm.

Note: true QLoRA (4-bit bitsandbytes) is not used here — bitsandbytes'
ROCm kernels aren't built for this GPU's architecture (gfx1201) yet and
crash on quantize. Plain bf16 LoRA is used instead, which a 3B model
comfortably fits in 16GB VRAM without needing quantization.

Loss is only computed on the target message (author + gap + text) that
comes after the "<|next|>" marker, not on the context window.
"""
import json
import os

# ROCm exposes the CPU as an extra "cuda" device alongside the real GPUs,
# which makes transformers.Trainer wrap the model in DataParallel across
# all of them (including the bogus CPU entry) and crash on NCCL. Pin to
# a single real GPU, before torch is imported.
#
# Multi-GPU DDP across both cards was tried and hangs indefinitely during
# RCCL process-group init (with or without NCCL_P2P_DISABLE) — these
# consumer RDNA cards have no direct GPU-to-GPU link (xGMI), and RCCL's
# fallback transport negotiation appears to deadlock on this ROCm build.
# Not worth chasing further; single-GPU is the reliable path here.
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import signal
from pathlib import Path

import torch
import transformers
transformers.logging.set_verbosity_info()
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

MODEL_PATH = "models/Qwen2.5-3B-Instruct"
DATASET_PATH = "finetune_dataset.jsonl"
OUTPUT_DIR = "lora_out"
MAX_LENGTH = 1024
DEVICE = "cuda:0"  # RX 9070 XT


def load_dataset(tokenizer):
    records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    def tokenize(example):
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(
            example["completion"] + tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    ds = Dataset.from_list(records)
    return ds.map(tokenize, remove_columns=["prompt", "completion"])


class GracefulShutdown:
    """Set by a SIGTERM/SIGINT handler; checked once per step so the trainer
    can finish the in-flight step and save a proper checkpoint (with
    optimizer/scheduler state) instead of being killed mid-write."""

    requested = False

    @classmethod
    def register(cls):
        def handler(signum, frame):
            print(f"\nReceived signal {signum}, will checkpoint and stop after this step...")
            cls.requested = True

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)


class LossPrinterCallback(TrainerCallback):
    """The default HF console logger doesn't reliably print in this
    transformers version, so log metrics explicitly on every on_log event."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_local_process_zero:
            print(f"LOG: {logs}", flush=True)


class SaveOnShutdownCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if GracefulShutdown.requested:
            control.should_save = True
            control.should_training_stop = True
        return control


class PaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def find_latest_checkpoint():
    out_dir = Path(OUTPUT_DIR)
    if not out_dir.exists():
        return None
    checkpoints = sorted(
        out_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    train_ds = load_dataset(tokenizer)
    print(f"{len(train_ds)} training examples")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,
        report_to=[],
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=PaddingCollator(tokenizer),
        callbacks=[LossPrinterCallback(), SaveOnShutdownCallback()],
    )

    GracefulShutdown.register()

    resume_from = find_latest_checkpoint()
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")

    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except Exception:
        print("Training crashed — attempting an emergency checkpoint save...")
        try:
            trainer.save_model(f"{OUTPUT_DIR}/crashed")
            print(f"Emergency checkpoint saved to {OUTPUT_DIR}/crashed")
        except Exception as save_error:
            print(f"Emergency checkpoint save also failed: {save_error}")
        raise

    if GracefulShutdown.requested:
        print("Stopped early due to shutdown signal; latest checkpoint is in "
              f"{OUTPUT_DIR}/checkpoint-{trainer.state.global_step}")
        return

    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"Saved LoRA adapter to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
