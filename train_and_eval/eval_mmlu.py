#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
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

MMLU_BIO_SUBSETS = [
    "anatomy",
    "college_biology",
    "high_school_biology",
    "medical_genetics",
]

USER_INSTRUCTION = (
    "Answer with the letter only (A, B, C, D, or E). "
    "Do NOT include any words, punctuation, or explanation. "
    "Output a single uppercase letter corresponding to your choice.\n\n"
)


@dataclass
class EvalResult:
    model_tag: str
    subset: str
    num_samples: int
    num_correct: int
    accuracy: float


# ------------------------
# CLI
# ------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate base+LoRA on MMLU biology subsets (Qwen3-8B)."
    )
    p.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HF model name or local path (e.g. Qwen/Qwen3-8B)",
    )
    p.add_argument(
        "--adapters_path",
        type=str,
        default="",
        help="LoRA adapter directory (optional). Leave empty/'none' to skip LoRA eval.",
    )
    p.add_argument(
        "--merge_adapters",
        action="store_true",
        help="Merge LoRA into base model before vLLM inference.",
    )

    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--shuffle_seed", type=int, default=0)

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--max_new_tokens", type=int, default=1)

    p.add_argument("--result_path", type=str, default="")
    p.add_argument("--output_path", type=str, default="")

    return p.parse_args()


# ------------------------
# LoRA merge
# ------------------------
def merge_lora_to_temp_dir(
    base_model: str,
    adapters_path: str,
) -> Tuple[tempfile.TemporaryDirectory, str]:
    if PeftModel is None:
        raise RuntimeError("peft required for LoRA merging.")

    os.makedirs("models", exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(dir="models/")
    out_path = temp_dir.name

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )

    model = PeftModel.from_pretrained(
        model=model,
        model_id=adapters_path,
        torch_dtype=torch.bfloat16,
    )
    model = model.merge_and_unload()

    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    del model, tokenizer
    torch.cuda.empty_cache()

    return temp_dir, out_path


# ------------------------
# Dataset — one MMLU subset
# ------------------------
def load_mmlu_subset(subset: str, args: argparse.Namespace) -> Dataset:
    ds = load_dataset("cais/mmlu", subset, split="test")

    if args.shuffle_seed:
        ds = ds.shuffle(seed=args.shuffle_seed)

    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    ds = ds.map(
        lambda x: {"text": FormatExtractor.qa_mc(x, eval_format_official=True)},
        load_from_cache_file=False,
    )

    return ds


# ------------------------
# Chat template formatting
# ------------------------
def format_prompts_for_qwen(tokenizer, prompts: List[str]) -> List[str]:
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": USER_INSTRUCTION + p}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Strip empty think block injected by Qwen3-Base tokenizer
        prompt = re.sub(r"<think>.*?</think>\s*", "", prompt, flags=re.DOTALL)
        formatted.append(prompt)
    return formatted


# ------------------------
# Evaluation — single subset
# ------------------------
def eval_subset_with_vllm(
    llm: "LLM",
    subset: str,
    prompts: List[str],
    choices: List[List[str]],
    gold: List[int],
    sampling: SamplingParams,
    model_tag: str,
) -> Tuple[EvalResult, List[int]]:

    tokenizer = llm.get_tokenizer()
    formatted_prompts = format_prompts_for_qwen(tokenizer, prompts)
    outputs = llm.generate(formatted_prompts, sampling)
    vocab = tokenizer.get_vocab()

    preds = c.get_classifications(outputs, choices, vocab, task="bioqa")
    preds = [p if p is not None else -1 for p in preds]

    print(f"  [{subset}] Prediction distribution: {dict(Counter(preds))}")

    total_samples = len(gold)
    num_correct = sum(int(p == g) for p, g in zip(preds, gold))
    accuracy = num_correct / total_samples if total_samples > 0 else 0.0

    result = EvalResult(
        model_tag=model_tag,
        subset=subset,
        num_samples=total_samples,
        num_correct=num_correct,
        accuracy=accuracy,
    )
    return result, preds


# ------------------------
# Evaluate all subsets with one LLM instance
# ------------------------
def eval_all_subsets(
    model_path: str,
    args: argparse.Namespace,
    sampling: SamplingParams,
) -> Tuple[Dict[str, EvalResult], Dict[str, List[int]], Dict[str, Dataset]]:

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        enforce_eager=True,
        swap_space=8,
        trust_remote_code=True,
    )

    all_results: Dict[str, EvalResult] = {}
    all_preds: Dict[str, List[int]] = {}
    all_datasets: Dict[str, Dataset] = {}

    for subset in MMLU_BIO_SUBSETS:
        print(f"\n--- Evaluating subset: {subset} ---")
        ds = load_mmlu_subset(subset, args)
        prompts = ds["text"]
        choices = ds["choices"]
        gold = ds["answer"]

        result, preds = eval_subset_with_vllm(
            llm=llm,
            subset=subset,
            prompts=prompts,
            choices=choices,
            gold=gold,
            sampling=sampling,
            model_tag=model_path,
        )
        all_results[subset] = result
        all_preds[subset] = preds
        all_datasets[subset] = ds

        print(
            f"  [{subset}] Acc: {result.accuracy:.6f} "
            f"({result.num_correct}/{result.num_samples})"
        )

    return all_results, all_preds, all_datasets


def print_summary(results: Dict[str, EvalResult], label: str) -> float:
    print(f"\n{'=' * 50}")
    print(f"Summary — {label}")
    print(f"{'=' * 50}")
    total_correct = 0
    total_samples = 0
    for subset, r in results.items():
        print(f"  {subset:<25} {r.accuracy:.4f}  ({r.num_correct}/{r.num_samples})")
        total_correct += r.num_correct
        total_samples += r.num_samples
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"  {'AVERAGE':<25} {avg_acc:.4f}  ({total_correct}/{total_samples})")
    print(f"{'=' * 50}\n")
    return avg_acc


# ------------------------
# Main
# ------------------------
def main() -> None:
    args = parse_args()
    t0 = time()

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        logprobs=20,  # increased to ensure A/B/C/D/E are always in the list
    )

    lora_results: Optional[Dict[str, EvalResult]] = None
    lora_preds: Optional[Dict[str, List[int]]] = None
    lora_datasets: Optional[Dict[str, Dataset]] = None
    temp_dir = None
    avg_acc = 0.0

    if args.adapters_path and args.adapters_path.lower() != "none":
        temp_dir, merged_path = merge_lora_to_temp_dir(
            args.base_model, args.adapters_path
        )
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        lora_results, lora_preds, lora_datasets = eval_all_subsets(
            model_path=merged_path,
            args=args,
            sampling=sampling,
        )
        avg_acc = print_summary(lora_results, label=f"LoRA ({args.adapters_path})")

    # ---- save results ----
    if args.result_path:
        if lora_results is None:
            print("Warning: no LoRA result to save (--adapters_path not set).")
        else:
            out = {
                "model": args.base_model,
                "adapters": args.adapters_path,
                "subsets": {
                    subset: {
                        "accuracy": r.accuracy,
                        "num_correct": r.num_correct,
                        "num_samples": r.num_samples,
                    }
                    for subset, r in lora_results.items()
                },
                "average_accuracy": avg_acc,
            }
            with open(args.result_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"Results saved to {args.result_path}")

    # ---- save outputs ----
    if args.output_path and lora_preds is not None:
        max_choices = ["A", "B", "C", "D", "E", "N/A"]
        with open(args.output_path, "w") as jsonl_file:
            for subset in MMLU_BIO_SUBSETS:
                ds = lora_datasets[subset]
                preds = lora_preds[subset]
                val_dataset = ds.map(
                    lambda x: {"letter_answer": max_choices[x["answer"]]}
                )
                letter_preds = [max_choices[p] for p in preds]
                for instruction, prediction, reference in zip(
                    ds["text"], letter_preds, val_dataset["letter_answer"]
                ):
                    sample = {
                        "subset": subset,
                        "instruction": instruction,
                        "prediction": prediction,
                        "reference": reference,
                    }
                    jsonl_file.write(json.dumps(sample) + "\n")
        print(f"Outputs saved to {args.output_path}")

    if temp_dir is not None:
        temp_dir.cleanup()

    print(f"\nTotal script time: {time() - t0:.2f}s")


if __name__ == "__main__":
    main()
