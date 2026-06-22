#!/usr/bin/env python3
"""
Zero-shot evaluation (no LoRA, no adapter merging).
Supports base models and instruct models directly.

Tasks
-----
bioqa  : ScienceQA biology test split        
medqa  : MedMCQA validation split            
csqa   : CommonsenseQA 2.0 validation split  

Usage
-----
# Instruct model, MedQA
python eval_instruct_multi.py --task medqa \
    --model Qwen/Qwen3-8B \
    --result_path results/medqa_instruct_zeroshot.json

# Base model, CSQA
python eval_instruct_multi.py --task csqa \
    --model Qwen/Qwen3-8B-Base \
    --result_path results/csqa_base_zeroshot.json

# BioQA
python eval_instruct_multi.py --task bioqa \
    --model Qwen/Qwen3-8B \
    --result_path results/bioqa_instruct_zeroshot.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from time import time
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

import common as c
from format_eval import FormatExtractor

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
    split: str
    max_samples: int  # paper's eval set size (0 = all)
    answer_labels: List[str]
    instruction: str
    format_fn: str


TASK_REGISTRY: Dict[str, TaskConfig] = {
    "bioqa": TaskConfig(
        name="bioqa",
        hf_dataset="derek-thomas/ScienceQA",
        hf_config=None,
        split="test",
        max_samples=397,
        answer_labels=["A", "B", "C", "D", "E"],
        instruction=MC_INSTRUCTION,
        format_fn="bioqa",
    ),
    "medqa": TaskConfig(
        name="medqa",
        hf_dataset="openlifescienceai/MedMCQA",
        hf_config=None,
        split="validation",
        max_samples=4183,
        answer_labels=["A", "B", "C", "D"],
        instruction=MC_INSTRUCTION,
        format_fn="medqa",
    ),
    "csqa": TaskConfig(
        name="csqa",
        hf_dataset="tasksource/commonsense_qa_2.0",
        hf_config=None,
        split="validation",
        max_samples=2541,
        answer_labels=["A", "B"],
        instruction=YESNO_INSTRUCTION,
        format_fn="csqa",
    ),
}


# ── Dataset formatters ────────────────────────────────────────────────────────


class TaskFormatter:
    """
    Returns (prompts, choices_list, gold_indices) for a loaded HF dataset.
    prompts      : List[str]        – raw question + choices (no template)
    choices_list : List[List[str]]  – raw option strings per sample
    gold_indices : List[int]        – 0-based index of correct answer
    """

    @staticmethod
    def bioqa(ds: Dataset) -> Tuple[List[str], List[List[str]], List[int]]:
        ds = ds.filter(lambda x: x.get("topic") == "biology" and not x.get("image"))
        ds = ds.map(
            lambda x: {"text": FormatExtractor.qa_mc(x, eval_format_official=True)},
            load_from_cache_file=False,
        )
        return ds["text"], ds["choices"], list(ds["answer"])

    @staticmethod
    def medqa(ds: Dataset) -> Tuple[List[str], List[List[str]], List[int]]:
        prompts, choices_list, gold = [], [], []
        for ex in ds:
            opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
            option_lines = "\n".join(
                f"{MAX_CHOICES[i]}. {t}" for i, t in enumerate(opts)
            )
            prompts.append(f"{ex['question']}\n{option_lines}")
            choices_list.append(opts)
            gold.append(int(ex["cop"]))
        return prompts, choices_list, gold

    @staticmethod
    def csqa(ds: Dataset) -> Tuple[List[str], List[List[str]], List[int]]:
        prompts, choices_list, gold = [], [], []
        for ex in ds:
            prompt = f"{ex['question']}\nA. Yes\nB. No"
            prompts.append(prompt)
            choices_list.append(["Yes", "No"])
            ans_raw = str(ex.get("answer", "")).strip().lower()
            gold.append(0 if ans_raw in {"true", "yes", "1"} else 1)
        return prompts, choices_list, gold


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-shot evaluation on BioQA / MedQA / CSQA."
    )
    p.add_argument(
        "--task",
        type=str,
        default="medqa",
        choices=list(TASK_REGISTRY.keys()),
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name or local path (base or instruct).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Override task default sample cap (0 = use task default).",
    )
    p.add_argument("--shuffle_seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--max_new_tokens", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--result_path", type=str, default="")
    p.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional JSONL file to save per-sample predictions.",
    )
    return p.parse_args()


# ── Prompt formatting ─────────────────────────────────────────────────────────


def format_prompts(tokenizer, prompts: List[str], instruction: str) -> List[str]:
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": "/no_think\n" + instruction + p}]
        chat = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        chat = re.sub(r"<think>.*?</think>\s*", "", chat, flags=re.DOTALL)
        chat += "<think>\n</think>\n"
        formatted.append(chat)
    return formatted


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    t0 = time()

    task_cfg = TASK_REGISTRY[args.task]

    # ── Load dataset ──────────────────────────────────────────────────────────
    load_kwargs = dict(split=task_cfg.split)
    if task_cfg.hf_config:
        load_kwargs["name"] = task_cfg.hf_config

    ds = load_dataset(task_cfg.hf_dataset, **load_kwargs)

    if args.shuffle_seed:
        ds = ds.shuffle(seed=args.shuffle_seed)

    cap = args.max_samples if args.max_samples > 0 else task_cfg.max_samples
    if cap > 0:
        ds = ds.select(range(min(cap, len(ds))))

    # ── Format ────────────────────────────────────────────────────────────────
    formatter = getattr(TaskFormatter, task_cfg.format_fn)
    prompts, choices, gold = formatter(ds)

    print(f"Model : {args.model}")
    print(f"Task  : {args.task.upper()}  |  samples: {len(prompts)}")

    # ── Load model ────────────────────────────────────────────────────────────
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        swap_space=8,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # ── Generate ──────────────────────────────────────────────────────────────
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        logprobs=20,
    )

    formatted = format_prompts(tokenizer, prompts, task_cfg.instruction)

    # Debug: first sample
    first_out = llm.generate([formatted[0]], sampling)[0]
    print(f"  [DEBUG] first generated text : {repr(first_out.outputs[0].text)}")

    outputs = llm.generate(formatted, sampling)

    vocab = tokenizer.get_vocab()
    preds = c.get_classifications(outputs, choices, vocab, task=task_cfg.name)
    preds = [p if p is not None else -1 for p in preds]

    print("Prediction distribution:", dict(Counter(preds)))

    # ── Score ─────────────────────────────────────────────────────────────────
    total = len(gold)
    correct = sum(int(p == g) for p, g in zip(preds, gold))
    acc = correct / total if total > 0 else 0.0

    print("\n" + "=" * 55)
    print("ZERO-SHOT EVALUATION RESULT")
    print("=" * 55)
    print(f"  {args.task.upper():<28} {acc:.4f}  ({correct}/{total})")
    print("=" * 55)

    # ── Save result JSON ──────────────────────────────────────────────────────
    if args.result_path:
        os.makedirs(os.path.dirname(args.result_path) or ".", exist_ok=True)
        out = {
            "model": args.model,
            "task": args.task,
            "accuracy": acc,
            "num_correct": correct,
            "num_samples": total,
        }
        with open(args.result_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved -> {args.result_path}")

    # ── Save per-sample predictions JSONL ─────────────────────────────────────
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        labels = task_cfg.answer_labels + ["N/A"]
        letter_preds = [labels[min(p, len(labels) - 1)] for p in preds]
        gold_letters = [labels[min(g, len(labels) - 1)] for g in gold]
        with open(args.output_path, "w") as f:
            for prompt, pred, ref in zip(prompts, letter_preds, gold_letters):
                f.write(
                    json.dumps(
                        {"instruction": prompt, "prediction": pred, "reference": ref}
                    )
                    + "\n"
                )
        print(f"Outputs saved -> {args.output_path}")

    print(f"\nTotal time: {time() - t0:.2f}s")


if __name__ == "__main__":
    main()
