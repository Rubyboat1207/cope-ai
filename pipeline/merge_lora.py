"""Merges a LoRA checkpoint into the base model weights, producing a
standalone HF model directory ready for GGUF conversion."""
import argparse
import os

os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/Qwen2.5-3B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to lora_out/checkpoint-N")
    parser.add_argument("--out", default="models/merged", help="output directory")
    args = parser.parse_args()

    print(f"Loading base model from {MODEL_PATH}...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading LoRA adapter from {args.checkpoint}...")
    model = PeftModel.from_pretrained(base, args.checkpoint)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.out}...")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
