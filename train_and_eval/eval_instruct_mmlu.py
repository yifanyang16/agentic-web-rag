#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from time import time
from typing import List

from datasets import load_dataset
from vllm import LLM, SamplingParams

MAX_CHOICES = ["A", "B", "C", "D", "E"]

USER_INSTRUCTION = (
    "/no_think\n"
    "Answer with the letter only (A, B, C, D, or E). "
    "Do NOT include any words, punctuation, or explanation. "
    "Output a single uppercase letter corresponding to your choice.\n\n"
)

MMLU_BIO_SUBSETS = [
    "anatomy",
    "college_biology",
    "high_school_biology",
    "medical_genetics",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--result_path", type=str, default="")
    return p.parse_args()


def format_prompts(tokenizer, texts: List[str]) -> List[str]:
    formatted = []
    for t in texts:
        messages = [{"role": "user", "content": USER_INSTRUCTION + t}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt = re.sub(r"<think>.*?</think>", "", prompt, flags=re.DOTALL)
        prompt += "<think>\n</think>\n"
        formatted.append(prompt)
    return formatted


def get_classifications(outputs, choices, vocab, num_options):
    max_choices = MAX_CHOICES[:num_options]
    logprob_idxs = [vocab[letter] for letter in max_choices]
    choices_idxs = [logprob_idxs[: len(choice)] for choice in choices]

    clf = []
    for output, idxs in zip(outputs, choices_idxs):
        if output.outputs[0].text == "":
            print("EMPTY STRING!!!")
            clf.append(-1)
            continue
        logprobs_idx = [output.outputs[0].logprobs[0].get(idx, None) for idx in idxs]
        logprobs = [x.logprob if x is not None else -float("inf") for x in logprobs_idx]
        if all(x == -float("inf") for x in logprobs):
            clf.append(-1)
        else:
            clf.append(logprobs.index(max(logprobs)))

    return clf


def evaluate(llm, tokenizer, sampling, texts, gold, choices, num_options, name) -> dict:
    prompts = format_prompts(tokenizer, texts)
    outputs = llm.generate(prompts, sampling)

    # debug: first sample only
    first = outputs[0].outputs[0]
    print(f"  [DEBUG] generated text : {repr(first.text)}")
    if first.logprobs:
        top5 = sorted(
            first.logprobs[0].items(), key=lambda x: x[1].logprob, reverse=True
        )[:5]
        print(
            f"  [DEBUG] top-5 tokens   : {[(tokenizer.decode([tid]), round(lp.logprob, 3)) for tid, lp in top5]}"
        )

    vocab = tokenizer.get_vocab()
    preds = get_classifications(outputs, choices, vocab, num_options)
    preds = [p if p is not None else -1 for p in preds]

    print("Prediction distribution:", dict(Counter(preds)))

    total = len(gold)
    correct = sum(int(p == g) for p, g in zip(preds, gold))
    acc = correct / total if total > 0 else 0.0
    print(f"  [{name}] Acc: {acc:.4f} ({correct}/{total})")
    return {"accuracy": acc, "num_correct": correct, "num_samples": total}


def eval_scienceqa(llm, tokenizer, sampling, args) -> dict:
    print("\n=== Evaluating: ScienceQA biology (test) ===")
    ds = load_dataset("derek-thomas/ScienceQA", split="test")
    ds = ds.filter(lambda x: x.get("topic") == "biology")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    texts = []
    for x in ds:
        q = (
            x["question"]
            + "\n"
            + "\n".join(f"{MAX_CHOICES[i]}. {c}" for i, c in enumerate(x["choices"]))
        )
        if x.get("hint"):
            q += f"\nHint: {x['hint']}"
        texts.append(q)

    num_options = max(len(x["choices"]) for x in ds)
    choices = [x["choices"] for x in ds]
    return evaluate(
        llm,
        tokenizer,
        sampling,
        texts,
        list(ds["answer"]),
        choices,
        num_options,
        "ScienceQA-biology",
    )


def eval_mmlu(llm, tokenizer, sampling, args) -> dict:
    print("\n=== Evaluating: MMLU biology subsets ===")
    subset_results = {}
    total_correct = total_samples = 0

    for subset in MMLU_BIO_SUBSETS:
        print(f"\n--- {subset} ---")
        ds = load_dataset("cais/mmlu", subset, split="test")
        if args.max_samples > 0:
            ds = ds.select(range(min(args.max_samples, len(ds))))

        texts = [
            x["question"]
            + "\n"
            + "\n".join(f"{MAX_CHOICES[i]}. {c}" for i, c in enumerate(x["choices"]))
            for x in ds
        ]
        choices = [x["choices"] for x in ds]
        r = evaluate(
            llm, tokenizer, sampling, texts, list(ds["answer"]), choices, 4, subset
        )
        subset_results[subset] = r
        total_correct += r["num_correct"]
        total_samples += r["num_samples"]

    avg = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"\n  MMLU Average Acc: {avg:.4f} ({total_correct}/{total_samples})")
    return {
        "subsets": subset_results,
        "average_accuracy": avg,
        "total_correct": total_correct,
        "total_samples": total_samples,
    }


def main():
    args = parse_args()
    t0 = time()

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=20,
    )

    scienceqa_result = eval_scienceqa(llm, tokenizer, sampling, args)
    mmlu_result = eval_mmlu(llm, tokenizer, sampling, args)

    print("\n" + "=" * 55)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 55)
    print(
        f"  {'ScienceQA biology':<28} {scienceqa_result['accuracy']:.4f}  ({scienceqa_result['num_correct']}/{scienceqa_result['num_samples']})"
    )
    print(f"  {'─' * 50}")
    for subset, r in mmlu_result["subsets"].items():
        print(
            f"  {subset:<28} {r['accuracy']:.4f}  ({r['num_correct']}/{r['num_samples']})"
        )
    print(f"  {'─' * 50}")
    print(
        f"  {'MMLU Average':<28} {mmlu_result['average_accuracy']:.4f}  ({mmlu_result['total_correct']}/{mmlu_result['total_samples']})"
    )
    print("=" * 55)

    if args.result_path:
        os.makedirs(os.path.dirname(args.result_path) or ".", exist_ok=True)
        out = {
            "model": args.model,
            "scienceqa_biology": scienceqa_result,
            "mmlu_biology": mmlu_result,
        }
        with open(args.result_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to: {args.result_path}")

    print(f"\nTotal time: {time() - t0:.2f}s")


if __name__ == "__main__":
    main()
