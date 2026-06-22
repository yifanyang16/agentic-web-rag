#!/usr/bin/env python3
"""
Multi-task evaluation: BioQA | MedQA | CSQA 

BioQA  : 397 test samples, ScienceQA biology subset 
MedQA  : 4 183 validation samples, MedMCQA
CSQA   : 2 541 validation samples, CommonsenseQA 2.0 (yes/no)
RecipeGen: skipped (generation task, no MC accuracy)

Usage
-----
# BioQA 
python eval_multi.py --task bioqa \
    --adapters_path checkpoints/lora_run1 \
    --result_path results/bioqa.json

# MedQA
python eval_multi.py --task medqa \
    --adapters_path checkpoints/lora_run1 \
    --result_path results/medqa.json

# CSQA (yes/no)
python eval_multi.py --task csqa \
    --adapters_path checkpoints/lora_run1 \
    --result_path results/csqa.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from time import time
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

import common as c
from format_eval import FormatExtractor

try:
    from peft.peft_model import PeftModel
except Exception:
    PeftModel = None


# ── Instruction strings ────────────────────────────────────────────────────────

MC_INSTRUCTION = (
    "Answer with the letter only (A, B, C, D). "
    "Do NOT include any words, punctuation, or explanation. "
    "Output a single uppercase letter corresponding to your choice.\n\n"
)

YESNO_INSTRUCTION = (
    "Answer with the letter only (A or B). "
    "Do NOT include any words, punctuation, or explanation. "
    "Output a single uppercase letter corresponding to your choice.\n\n"
)

# ── Task registry ──────────────────────────────────────────────────────────────


@dataclass
class TaskConfig:
    name: str
    hf_dataset: str
    hf_config: Optional[str]
    split: str
    max_samples: int
    answer_labels: List[str]
    instruction: str
    format_fn: str


TASK_REGISTRY: Dict[str, TaskConfig] = {
    "bioqa": TaskConfig(
        name="bioqa",
        hf_dataset="derek-thomas/ScienceQA",
        hf_config=None,
        split="test",
        max_samples=800,
        answer_labels=["A", "B", "C", "D"],
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
    @staticmethod
    def bioqa(ds: Dataset) -> Tuple[List[str], List[List[str]], List[int]]:
        ds = ds.filter(lambda x: x.get("topic") == "biology")
        ds = ds.map(
            lambda x: {"text": FormatExtractor.qa_mc(x, eval_format_official=True)},
            load_from_cache_file=False,
        )
        prompts = ds["text"]
        choices = ds["choices"]
        gold = ds["answer"]
        return prompts, choices, gold

    @staticmethod
    def medqa(ds: Dataset) -> Tuple[List[str], List[List[str]], List[int]]:
        prompts, choices_list, gold = [], [], []
        for ex in ds:
            opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
            labels = ["A", "B", "C", "D"]
            option_lines = "\n".join(f"{l}. {t}" for l, t in zip(labels, opts))
            prompt = f"{ex['question']}\n{option_lines}"
            prompts.append(prompt)
            choices_list.append(opts)
            gold.append(int(ex["cop"]))
        return prompts, choices_list, gold

    @staticmethod
    def csqa(ds: Dataset) -> Tuple[List[str], List[List[str]], List[int]]:
        prompts, choices_list, gold = [], [], []
        for ex in ds:
            q = ex["question"]
            opts = ["Yes", "No"]
            prompt = f"{q}\nA. Yes\nB. No"
            prompts.append(prompt)
            choices_list.append(opts)
            ans_raw = str(ex.get("answer", "")).strip().lower()
            gold.append(0 if ans_raw in {"true", "yes", "1"} else 1)
        return prompts, choices_list, gold


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate base+LoRA on BioQA / MedQA / CSQA."
    )
    p.add_argument(
        "--task",
        type=str,
        default="bioqa",
        choices=list(TASK_REGISTRY.keys()),
    )
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--adapters_path", type=str, default="")
    p.add_argument("--merge_adapters", action="store_true")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--shuffle_seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--max_new_tokens", type=int, default=1)
    p.add_argument("--result_path", type=str, default="")
    p.add_argument("--output_path", type=str, default="")
    return p.parse_args()


# ── LoRA merge ────────────────────────────────────────────────────────────────


def merge_lora_to_temp_dir(
    base_model: str,
    adapters_path: str,
) -> Tuple[tempfile.TemporaryDirectory, str]:
    if PeftModel is None:
        raise RuntimeError("peft is required for LoRA merging.")

    os.makedirs("models", exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(dir="models/")
    out_path = temp_dir.name

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(
        model=model, model_id=adapters_path, torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload()
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    del model, tokenizer
    torch.cuda.empty_cache()
    return temp_dir, out_path


# ── Prompt formatting ─────────────────────────────────────────────────────────


def format_prompts_for_model(
    tokenizer, prompts: List[str], instruction: str
) -> List[str]:
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": instruction + p}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt = re.sub(r"<think>.*?</think>\s*", "", prompt, flags=re.DOTALL)
        formatted.append(prompt)
    return formatted


# ── Evaluation ────────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    model_tag: str
    task: str
    num_samples: int
    num_correct: int
    accuracy: float


def eval_with_vllm(
    model_path: str,
    task_cfg: TaskConfig,
    prompts: List[str],
    choices: List[List[str]],
    gold: List[int],
    sampling: SamplingParams,
) -> Tuple[EvalResult, List[int]]:

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        enforce_eager=True,
        swap_space=8,
        trust_remote_code=True,
    )

    tokenizer = llm.get_tokenizer()
    formatted = format_prompts_for_model(tokenizer, prompts, task_cfg.instruction)
    outputs = llm.generate(formatted, sampling)

    vocab = tokenizer.get_vocab()
    preds = c.get_classifications(outputs, choices, vocab, task=task_cfg.name)
    preds = [p if p is not None else -1 for p in preds]

    print("Prediction distribution:", dict(Counter(preds)))

    total = len(gold)
    correct = sum(int(p == g) for p, g in zip(preds, gold))

    result = EvalResult(
        model_tag=model_path,
        task=task_cfg.name,
        num_samples=total,
        num_correct=correct,
        accuracy=correct / total if total > 0 else 0.0,
    )
    return result, preds


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

    # ── Format first, then cap ────────────────────────────────────────────────
    # IMPORTANT: filter (e.g. biology+no image for bioqa) happens inside formatter.
    # We must cap AFTER formatting so the filter is applied to the full dataset first.
    formatter = getattr(TaskFormatter, task_cfg.format_fn)
    prompts, choices, gold = formatter(ds)

    cap = args.max_samples if args.max_samples > 0 else task_cfg.max_samples
    if cap > 0:
        prompts = prompts[:cap]
        choices = choices[:cap]
        gold = gold[:cap]

    print(f"Task: {args.task.upper()}  |  samples: {len(prompts)}")

    # ── Sampling params ───────────────────────────────────────────────────────
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        logprobs=20,
    )

    # ── LoRA eval ─────────────────────────────────────────────────────────────
    lora_res: Optional[EvalResult] = None
    lora_preds: Optional[List[int]] = None
    temp_dir = None

    if args.adapters_path and args.adapters_path.lower() != "none":
        temp_dir, merged_path = merge_lora_to_temp_dir(
            args.base_model, args.adapters_path
        )
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        lora_res, lora_preds = eval_with_vllm(
            model_path=merged_path,
            task_cfg=task_cfg,
            prompts=prompts,
            choices=choices,
            gold=gold,
            sampling=sampling,
        )
        print(
            f"[{args.task.upper()} / LoRA] Acc: {lora_res.accuracy:.4f} "
            f"({lora_res.num_correct}/{lora_res.num_samples})"
        )

    # ── Save result JSON ──────────────────────────────────────────────────────
    if args.result_path:
        if lora_res is None:
            print("Warning: no LoRA result to save (--adapters_path not set).")
        else:
            out = {
                "task": args.task,
                "model": args.base_model,
                "adapters": args.adapters_path,
                "accuracy": lora_res.accuracy,
                "num_correct": lora_res.num_correct,
                "num_samples": lora_res.num_samples,
            }
            with open(args.result_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"Results saved -> {args.result_path}")

    # ── Save per-sample predictions JSONL ─────────────────────────────────────
    if args.output_path:
        if lora_preds is None:
            print("Warning: no LoRA predictions to save (--adapters_path not set).")
        else:
            labels = task_cfg.answer_labels + ["N/A"]
            letter_preds = [labels[min(p, len(labels) - 1)] for p in lora_preds]
            gold_letters = [labels[min(g, len(labels) - 1)] for g in gold]

            with open(args.output_path, "w") as f:
                for prompt, pred, ref in zip(prompts, letter_preds, gold_letters):
                    f.write(
                        json.dumps(
                            {
                                "instruction": prompt,
                                "prediction": pred,
                                "reference": ref,
                            }
                        )
                        + "\n"
                    )
            print(f"Outputs saved -> {args.output_path}")

    if temp_dir is not None:
        temp_dir.cleanup()

    print(f"\nTotal script time: {time() - t0:.2f}s")


if __name__ == "__main__":
    main()
