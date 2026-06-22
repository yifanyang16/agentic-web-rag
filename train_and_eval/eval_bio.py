# Example:
#   python eval_indomian.py \
#     --base_model Qwen/Qwen3-8B-Base \
#     --adapters_path runs/qwen3_scienceqa_bio \
#     --result_path runs/qwen3_scienceqa_bio/eval_results.json

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import tempfile
from collections import Counter
from time import time
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import common as c
from format_eval import FormatExtractor

MMLU_BIO_SUBSETS = [
    "anatomy",
    "college_biology",
    "high_school_biology",
    "medical_genetics",
]

MAX_CHOICES = ["A", "B", "C", "D", "E"]
LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

USER_INSTRUCTION = (
    "Answer with the letter only (A, B, C, D, or E). "
    "Do NOT include any words, punctuation, or explanation. "
    "Output a single uppercase letter corresponding to your choice.\n\n"
)


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B-Base")
    p.add_argument("--adapters_path", type=str, required=True)
    p.add_argument("--result_path", type=str, default="")
    p.add_argument("--eval_max_samples", type=int, default=0)
    p.add_argument("--eval_max_new_tokens", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return p.parse_args()


# ─────────────────────────────────────────
# Merge LoRA
# ─────────────────────────────────────────
def merge_lora(
    base_model: str, adapters_path: str
) -> Tuple[tempfile.TemporaryDirectory, str]:
    from peft import PeftModel

    os.makedirs("models", exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(dir="models/")
    out_path = temp_dir.name

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    print(f"Loading adapter: {adapters_path}")
    model = PeftModel.from_pretrained(
        model=model, model_id=adapters_path, torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload()
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    import time

    time.sleep(3)

    return temp_dir, out_path


# ─────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────
def apply_template(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = re.sub(r"<think>.*?</think>" + r"\s*", "", prompt, flags=re.DOTALL)
    return prompt


def make_scienceqa_prompt(sample, tokenizer) -> str:
    question_and_choices = FormatExtractor.qa_mc(sample, eval_format_official=True)
    return apply_template(tokenizer, USER_INSTRUCTION + question_and_choices)


def make_mmlu_prompt(sample, tokenizer) -> str:
    question = sample["question"]
    choices = sample["choices"]
    answers = "\n".join([f"{MAX_CHOICES[i]}. {text}" for i, text in enumerate(choices)])
    return apply_template(tokenizer, USER_INSTRUCTION + question + "\n" + answers)


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────
def eval_scienceqa(llm, tokenizer, args) -> dict:
    print("\n=== Evaluating: ScienceQA biology (test) ===")
    ds = load_dataset("derek-thomas/ScienceQA", split="test")
    ds = ds.filter(lambda x: x.get("topic") == "biology")
    if args.eval_max_samples > 0:
        ds = ds.select(range(min(args.eval_max_samples, len(ds))))

    formatted = [make_scienceqa_prompt(x, tokenizer) for x in ds]
    vocab = tokenizer.get_vocab()

    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        top_k=40,
        max_tokens=args.eval_max_new_tokens,
        logprobs=5,
    )
    outputs = llm.generate(formatted, sampling)
    preds = c.get_classifications(outputs, ds["choices"], vocab, task="bioqa")
    preds = [p if p is not None else -1 for p in preds]

    print(f"  Prediction distribution: {dict(Counter(preds))}")
    total = len(ds["answer"])
    correct = sum(int(p == g) for p, g in zip(preds, ds["answer"]))
    acc = correct / total if total > 0 else 0.0
    print(f"  ScienceQA Acc: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc, "num_correct": correct, "num_samples": total}


def eval_mmlu(llm, tokenizer, args) -> dict:
    print("\n=== Evaluating: MMLU biology subsets ===")
    vocab = tokenizer.get_vocab()

    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        top_k=40,
        max_tokens=args.eval_max_new_tokens,
        logprobs=5,
    )

    subset_results = {}
    total_correct = 0
    total_samples = 0

    for subset in MMLU_BIO_SUBSETS:
        print(f"\n--- {subset} ---")
        ds = load_dataset("cais/mmlu", subset, split="test")
        if args.eval_max_samples > 0:
            ds = ds.select(range(min(args.eval_max_samples, len(ds))))

        formatted = [make_mmlu_prompt(x, tokenizer) for x in ds]
        outputs = llm.generate(formatted, sampling)
        preds = c.get_classifications(outputs, ds["choices"], vocab, task="bioqa")
        preds = [p if p is not None else -1 for p in preds]

        gold = list(ds["answer"])

        print(f"  Prediction distribution: {dict(Counter(preds))}")
        total = len(gold)
        correct = sum(int(p == g) for p, g in zip(preds, gold))
        acc = correct / total if total > 0 else 0.0
        print(f"  [{subset}] Acc: {acc:.4f} ({correct}/{total})")

        subset_results[subset] = {
            "accuracy": acc,
            "num_correct": correct,
            "num_samples": total,
        }
        total_correct += correct
        total_samples += total

    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"\n  MMLU Average Acc: {avg_acc:.4f} ({total_correct}/{total_samples})")
    return {
        "subsets": subset_results,
        "average_accuracy": avg_acc,
        "total_correct": total_correct,
        "total_samples": total_samples,
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main() -> None:
    args = parse_args()
    t0 = time()

    # Merge LoRA into base model
    print("Merging LoRA adapter...")
    temp_dir, merged_path = merge_lora(args.base_model, args.adapters_path)

    try:
        from vllm import LLM

        llm = LLM(
            model=merged_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=1024,
            enforce_eager=True,
            swap_space=8,
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()

        scienceqa_result = eval_scienceqa(llm, tokenizer, args)
        mmlu_result = eval_mmlu(llm, tokenizer, args)

        # Summary
        print("\n" + "=" * 55)
        print("FINAL EVALUATION SUMMARY")
        print("=" * 55)
        print(
            f"  {'ScienceQA biology':<28} {scienceqa_result['accuracy']:.4f}  "
            f"({scienceqa_result['num_correct']}/{scienceqa_result['num_samples']})"
        )
        print(f"  {'─' * 50}")
        for subset, r in mmlu_result["subsets"].items():
            print(
                f"  {subset:<28} {r['accuracy']:.4f}  "
                f"({r['num_correct']}/{r['num_samples']})"
            )
        print(f"  {'─' * 50}")
        print(
            f"  {'MMLU Average':<28} {mmlu_result['average_accuracy']:.4f}  "
            f"({mmlu_result['total_correct']}/{mmlu_result['total_samples']})"
        )
        print("=" * 55)

        # Save results
        if args.result_path:
            out = {
                "base_model": args.base_model,
                "adapter": args.adapters_path,
                "scienceqa_biology": scienceqa_result,
                "mmlu_biology": mmlu_result,
            }
            os.makedirs(os.path.dirname(args.result_path) or ".", exist_ok=True)
            with open(args.result_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"\nResults saved to: {args.result_path}")

    finally:
        temp_dir.cleanup()

    print(f"\nTotal time: {time() - t0:.2f}s")


if __name__ == "__main__":
    main()
