#!/usr/bin/env python3
"""
Multi-task LoRA fine-tuning: MedQA | CSQA

MedQA : MedMCQA train split  (182 374 samples, capped via --max_train_samples)
CSQA  : CommonsenseQA 2.0 train split (9 260 samples, yes/no binary)

Usage
-----
# MedQA
python train_lindomain_multi.py --task medqa \
    --model_name_or_path Qwen/Qwen3-8B \
    --output_dir runs/medqa_lora \
    --batch_size 1 --grad_accum 16 --epochs 3 --lr 2e-4 --bf16 \
    --gradient_checkpointing

# CSQA
python train_indomain_multi.py --task csqa \
    --model_name_or_path Qwen/Qwen3-8B \
    --output_dir runs/csqa_lora \
    --batch_size 1 --grad_accum 16 --epochs 3 --lr 2e-4 --bf16 \
    --gradient_checkpointing
"""

from __future__ import annotations

import argparse
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

# ── Instruction strings ────────────────────────────────────────────────────────

MC_INSTRUCTION = (
    "Answer with the letter only (A, B, C, D, or E). "
    "Do NOT include any words, punctuation, or explanation. "
    "Output a single uppercase letter corresponding to your choice.\n\n"
)

YESNO_INSTRUCTION = (
    "Answer with the letter only (A or B). "
    "Do NOT include any words, punctuation, or explanation. "
    "Output a single uppercase letter corresponding to your choice.\n\n"
)

MAX_CHOICES = ["A", "B", "C", "D", "E"]


# ── Task registry ──────────────────────────────────────────────────────────────


@dataclass
class TaskConfig:
    name: str
    hf_dataset: str
    hf_config: Optional[str]
    train_split: str
    eval_split: str
    instruction: str
    format_fn: str
    answer_labels: List[str]


TASK_REGISTRY: Dict[str, TaskConfig] = {
    "medqa": TaskConfig(
        name="medqa",
        hf_dataset="openlifescienceai/MedMCQA",
        hf_config=None,
        train_split="train",
        eval_split="validation",
        instruction=MC_INSTRUCTION,
        format_fn="medqa",
        answer_labels=["A", "B", "C", "D"],
    ),
    "csqa": TaskConfig(
        name="csqa",
        hf_dataset="tasksource/commonsense_qa_2.0",
        hf_config=None,
        train_split="train",
        eval_split="validation",
        instruction=YESNO_INSTRUCTION,
        format_fn="csqa",
        answer_labels=["A", "B"],
    ),
}


# ── Dataset formatters ────────────────────────────────────────────────────────


class TaskFormatter:
    @staticmethod
    def medqa(ds: Dataset) -> Tuple[List[str], List[str]]:
        prompts, gold_letters = [], []
        label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        for ex in ds:
            opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
            option_lines = "\n".join(
                f"{MAX_CHOICES[i]}. {t}" for i, t in enumerate(opts)
            )
            prompts.append(f"{ex['question']}\n{option_lines}")
            gold_letters.append(label_map[int(ex["cop"])])
        return prompts, gold_letters

    @staticmethod
    def csqa(ds: Dataset) -> Tuple[List[str], List[str]]:
        prompts, gold_letters = [], []
        for ex in ds:
            prompt = f"{ex['question']}\nA. Yes\nB. No"
            prompts.append(prompt)
            ans_raw = str(ex.get("answer", "")).strip().lower()
            gold_letters.append("A" if ans_raw in {"true", "yes", "1"} else "B")
        return prompts, gold_letters


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for MedQA / CSQA.")

    p.add_argument(
        "--task", type=str, default="medqa", choices=list(TASK_REGISTRY.keys())
    )

    # I/O
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--output_dir", type=str, required=True)

    # Data
    p.add_argument("--max_train_samples", type=int, default=1000)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--eval_split", type=float, default=0.0)
    p.add_argument("--shuffle", action="store_true")

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
    )

    return p.parse_args()


# ── Prompt builder ────────────────────────────────────────────────────────────


def make_training_text(
    prompt: str, gold_letter: str, instruction: str, tokenizer
) -> str:
    user_content = instruction + prompt
    messages = [{"role": "user", "content": user_content}]
    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    chat = re.sub(r"<think>.*?</think>\s*", "", chat, flags=re.DOTALL)
    return chat + gold_letter


# ── Training ──────────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    task_cfg = TASK_REGISTRY[args.task]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading dataset: {task_cfg.hf_dataset} [{task_cfg.train_split}]")
    load_kwargs = dict(split=task_cfg.train_split)
    if task_cfg.hf_config:
        load_kwargs["name"] = task_cfg.hf_config
    raw_ds = load_dataset(task_cfg.hf_dataset, **load_kwargs)

    formatter = getattr(TaskFormatter, task_cfg.format_fn)
    prompts, gold_letters = formatter(raw_ds)

    texts = [
        make_training_text(p, g, task_cfg.instruction, tokenizer)
        for p, g in zip(prompts, gold_letters)
    ]
    ds = Dataset.from_dict({"text": texts})

    if args.max_train_samples > 0:
        ds = ds.shuffle(seed=args.seed)
        ds = ds.select(range(min(args.max_train_samples, len(ds))))
    elif args.shuffle:
        ds = ds.shuffle(seed=args.seed)

    print("=" * 60)
    print(f"Task: {args.task.upper()}  |  training samples: {len(ds)}")
    print("Sanity check — first training sample:")
    print(ds[0]["text"])
    print("=" * 60)

    eval_ds = None
    train_ds = ds
    if args.eval_split and args.eval_split > 0.0:
        split = train_ds.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one: --bf16 or --fp16.")

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


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
