import argparse
import json
import os
from typing import List, Dict, Any

from datasets import Dataset
from rapidfuzz import fuzz
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from create_prompt import build_prompt, parse_model_output


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--dedup-ratio", type=float, default=92.0)  # 放宽
    parser.add_argument("--task", type=str, default="bioqa_mc")
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--save_hf", action="store_true")

    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: str, data: List[Dict[str, Any]]):
    with open(path, "w") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def canonical_string(sample: Dict[str, Any]) -> str:
    question = sample.get("question", "")
    options = " ".join(sample.get("options", []))
    answer = sample.get("answer", "")
    return "\n".join([question, options, answer])


def dedup_among_samples(
    samples: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    kept = []
    for s in samples:
        s_str = canonical_string(s)
        if any(
            fuzz.token_set_ratio(s_str, canonical_string(t)) >= threshold for t in kept
        ):
            continue
        kept.append(s)
    return kept


def init_llm(model_name, temperature, top_p, top_k, max_new_tokens):
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
    )
    return llm, sampling_params


def format_prompt_for_qwen(prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prompt_too_long(formatted_prompt: str, tokenizer, max_tokens: int) -> bool:
    return len(tokenizer.encode(formatted_prompt)) > max_tokens


def to_final_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    base = {}
    for key in ("question", "options", "answer"):
        if key in sample:
            base[key] = sample[key]
    return base


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tag = args.task
    raw_path = os.path.join(args.output_dir, f"raw_generations_{tag}.jsonl")
    clean_path = os.path.join(args.output_dir, f"clean_samples_{tag}.jsonl")
    clean_raw_path = os.path.join(args.output_dir, f"clean_raw_generations_{tag}.jsonl")
    error_log_path = os.path.join(args.output_dir, f"format_errors_{tag}.csv")
    hf_path = os.path.join(args.output_dir, f"hf_dataset_{tag}")
    final_json_path = os.path.join(args.output_dir, f"final_samples_{tag}.jsonl")

    # ── Load corpus (already chunked by pipeline) ──────────────────────────────
    corpus = read_jsonl(args.corpus_path)
    passages = [ex["text"] for ex in corpus if ex.get("text", "").strip()]
    print(f"Loaded {len(passages)} passages from corpus.")

    # ── Init model and tokenizer ───────────────────────────────────────────────
    llm, sampling_params = init_llm(
        args.model, args.temperature, args.top_p, args.top_k, args.max_new_tokens
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=True
    )

    # ── Build and filter prompts ───────────────────────────────────────────────
    raw_prompts = []
    formatted_prompts = []

    for passage in passages:
        prompt = build_prompt(task=args.task, corpus_text=passage)
        formatted = format_prompt_for_qwen(prompt, tokenizer)
        if prompt_too_long(formatted, tokenizer, args.max_prompt_tokens):
            continue
        raw_prompts.append(prompt)
        formatted_prompts.append(formatted)

    print(f"Built {len(formatted_prompts)} prompts.")

    # ── Generate ───────────────────────────────────────────────────────────────
    results = llm.generate(formatted_prompts, sampling_params=sampling_params)

    raw_records = []
    for prompt, out in zip(raw_prompts, results):
        raw_records.append({"prompt": prompt, "raw_output": out.outputs[0].text})

    write_jsonl(raw_path, raw_records)
    print(f"Saved raw generations -> {raw_path}")

    # ── Parse and validate ─────────────────────────────────────────────────────
    valid = []
    error_lines = ["index,error\n"]

    for idx, rec in enumerate(raw_records):
        try:
            sample = parse_model_output(args.task, rec["raw_output"])
            sample["_raw_output"] = rec["raw_output"]
            sample["_prompt"] = rec["prompt"]
            valid.append(sample)
        except Exception as e:
            error_lines.append(f"{idx},{repr(e)}\n")

    with open(error_log_path, "w") as f:
        f.writelines(error_lines)

    print(f"{len(valid)} / {len(raw_records)} valid samples after parsing.")

    # ── Deduplication ──────────────────────────────────────────────────────────
    after_dedup = dedup_among_samples(valid, args.dedup_ratio)
    print(f"{len(after_dedup)} samples after deduplication.")

    # ── Write aligned clean_raw_generations ───────────────────────────────────
    write_jsonl(
        clean_raw_path,
        [{"prompt": s["_prompt"], "raw_output": s["_raw_output"]} for s in after_dedup],
    )

    # ── Strip private fields and write clean_samples ───────────────────────────
    for s in after_dedup:
        s.pop("_raw_output", None)
        s.pop("_prompt", None)

    write_jsonl(clean_path, after_dedup)
    print(f"Saved {len(after_dedup)} clean samples -> {clean_path}")

    # ── Build final dataset ────────────────────────────────────────────────────
    final = [to_final_record(s) for s in after_dedup]

    write_jsonl(final_json_path, final)
    print(f"Saved final samples ({len(final)}) -> {final_json_path}")

    if args.save_hf:
        Dataset.from_list(final).save_to_disk(hf_path)
        print(f"Saved HF dataset -> {hf_path}")


if __name__ == "__main__":
    main()
