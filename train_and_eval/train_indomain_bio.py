#!/usr/bin/env python3
# Example:
#   python train_lora_scienceqa.py \
#     --model_name_or_path Qwen/Qwen3-8B-Base \
#     --output_dir runs/qwen_scienceqa \
#     --batch_size 1 --grad_accum 16 --epochs 5 --lr 2e-4 --bf16 --gradient_checkpointing \
#     --max_seq_length 512

from __future__ import annotations

import argparse
import re
import os
from pathlib import Path
from typing import List

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

USER_INSTRUCTION = (
    "Answer with the letter only (A, B, C, D, or E). "
    "Do NOT include any words, punctuation, or explanation. "
    "Output a single uppercase letter corresponding to your choice.\n\n"
)

MAX_CHOICES = ["A", "B", "C", "D", "E"]


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # I/O
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-8B-Base")
    p.add_argument("--output_dir", type=str, required=True)

    # Training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)

    # Precision / memory
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Data
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Cap training set size (0 = all). Samples are shuffled before selection.",
    )
    p.add_argument("--eval_split", type=float, default=0.0)
    p.add_argument("--shuffle", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument(
        "--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"]
    )
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    return p.parse_args()


# ─────────────────────────────────────────
# Format helpers
# ─────────────────────────────────────────
def format_qa_mc(sample) -> str:
    """Format question + choices into a string."""
    choices = sample.get("choices", [])
    option_lines = "\n".join(
        f"{MAX_CHOICES[i]}. {text}" for i, text in enumerate(choices)
    )
    return f"{sample['question']}\n{option_lines}"


def make_training_text(sample, tokenizer) -> str:
    question_and_choices = format_qa_mc(sample)
    user_content = USER_INSTRUCTION + question_and_choices
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = re.sub(r"<think>.*?</think>\s*", "", prompt, flags=re.DOTALL)
    answer_letter = MAX_CHOICES[sample["answer"]]
    return prompt + answer_letter


# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────
def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = load_dataset("derek-thomas/ScienceQA", split="train")
    ds = ds.filter(lambda x: x.get("topic") == "biology")
    ds = ds.map(
        lambda x: {"text": make_training_text(x, tokenizer)},
        load_from_cache_file=False,
    )

    print("=" * 60)
    print("Sanity check — first training sample:")
    print(ds[0]["text"])
    print("=" * 60)

    if args.max_train_samples > 0:
        ds = ds.shuffle(seed=args.seed)
        ds = ds.select(range(min(args.max_train_samples, len(ds))))
        print(
            f"Using {len(ds)} training samples (max_train_samples={args.max_train_samples})"
        )
    elif args.shuffle:
        ds = ds.shuffle(seed=args.seed)

    eval_ds = None
    train_ds = ds
    if args.eval_split and args.eval_split > 0.0:
        split = train_ds.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one: --bf16 or --fp16 (not both).")

    torch_dtype = (
        torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    target_modules = [
        m.strip() for m in args.lora_target_modules.split(",") if m.strip()
    ]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.peft_config["default"].base_model_name_or_path = args.model_name_or_path

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable params: {len(trainable)}")
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters!")
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        eval_strategy="steps" if args.eval_split > 0.0 else "no",
        eval_steps=args.save_steps if args.eval_split > 0.0 else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )
    trainer.train()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved LoRA adapter to: {out_dir.resolve()}")

    return str(out_dir)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
