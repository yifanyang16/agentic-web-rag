# Example:
#   python train_lora_qwen.py \
#     --model_name_or_path Qwen/Qwen3-8B \
#     --dataset_dir bio_*.jsonl \
#     --output_dir runs/qwen3_8b_lora \
#     --batch_size 1 --grad_accum 16 --epochs 1 --lr 2e-4 --bf16 --gradient_checkpointing \
#     --max_seq_length 1024

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # I/O
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # Training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=3.0)
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
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--eval_split", type=float, default=0.0)
    p.add_argument("--shuffle", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"]
    )
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names",
    )

    return p.parse_args()


def craft_to_qwen(example: dict, tokenizer) -> str:
    question = example["question"]
    options = "\n".join(example["options"])
    answer = example["answer"]

    instruction = f"{question}\n{options}"

    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = re.sub(r"<think>.*?</think>\s*", "", prompt, flags=re.DOTALL)
    return prompt + answer


def make_formatting_func(tokenizer):
    def formatting_func(example):
        questions = (
            example["question"]
            if isinstance(example["question"], list)
            else [example["question"]]
        )
        options_list = (
            example["options"]
            if isinstance(example["options"][0], list)
            else [example["options"]]
        )
        answers = (
            example["answer"]
            if isinstance(example["answer"], list)
            else [example["answer"]]
        )

        return [
            craft_to_qwen({"question": q, "options": o, "answer": a}, tokenizer)
            for q, o, a in zip(questions, options_list, answers)
        ]

    return formatting_func


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.environ["TOKENIZER_PARALLELISM"] = "false"

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path.resolve()}")

    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    if "question" not in ds.column_names:
        raise ValueError(
            f"`question` column not found in dataset. Columns: {ds.column_names}"
        )

    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)

    eval_ds = None
    train_ds = ds
    if args.eval_split and args.eval_split > 0.0:
        split = train_ds.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Sanity check
    print("=" * 60)
    print("Sanity check — first training sample after reformatting:")
    print(make_formatting_func(tokenizer)(train_ds[0]))
    print("=" * 60)

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

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

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
        fp16=args.fp16 and not args.bf16,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        formatting_func=make_formatting_func(tokenizer),
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    trainer.train()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(f"Saved LoRA adapter to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
