"""
python attribution_judge.py \
    --input matched_bioqa.jsonl \
    --output attributed_bioqa.jsonl \
    --output-dir attributed_outputs \
    --tag bioqa \
    --model Qwen/Qwen2.5-14B-Instruct
"""

import json
import re
import argparse
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """You are an expert at evaluating the knowledge source of a multiple-choice question's correct answer.

Given a passage and a question with its correct answer, classify the source of the correct answer as exactly one of:

(A) Fully grounded in passage: the correct answer can be fully derived from the passage alone.
(B) Partially grounded in passage + parametric: the passage provides some relevant information, but the model's parametric knowledge is also required.
(C) Parametric only: the passage does not contain sufficient information; the answer relies entirely on the model's internal knowledge.

Respond with ONLY a JSON object in this format:
{"label": "A", "justification": "one sentence explanation"}"""

USER_TEMPLATE = """Passage:
{passage}

Question: {question}
Correct Answer: {answer_letter}. {answer_text}

Classify the knowledge source of the correct answer."""


def extract_passage(prompt: str) -> str:
    match = re.search(r"New passage:\s*(.*?)(?:\nNow create|$)", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    parts = prompt.split("New passage:")
    if len(parts) > 1:
        return parts[-1].split("Now create")[0].strip()
    match2 = re.search(r"Passage:\s*(.*?)(?:\nNow create|$)", prompt, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    return ""


def get_answer_text(options: list, answer_letter: str) -> str:
    for opt in options:
        stripped = opt.strip()
        if stripped.startswith(f"{answer_letter}.") or stripped.startswith(
            f"{answer_letter} ."
        ):
            return re.sub(r"^[A-E]\s*\.\s*", "", stripped).strip()
    idx = ord(answer_letter) - ord("A")
    if 0 <= idx < len(options):
        return re.sub(r"^[A-E]\s*\.\s*", "", options[idx].strip()).strip()
    return ""


def build_chat_prompt(tokenizer, passage, question, answer_letter, answer_text):
    user_msg = USER_TEMPLATE.format(
        passage=passage,
        question=question,
        answer_letter=answer_letter,
        answer_text=answer_text,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_judge_output(raw: str) -> dict:
    raw_clean = re.sub(
        r"^```json\s*|^```\s*|```$", "", raw.strip(), flags=re.MULTILINE
    ).strip()
    raw_clean = re.sub(r"<think>.*?</think>", "", raw_clean, flags=re.DOTALL).strip()
    return json.loads(raw_clean)


def compute_metrics(dist: dict) -> dict:
    a = dist.get("A", 0)
    b = dist.get("B", 0)
    c = dist.get("C", 0)
    valid = a + b + c
    par = (a + b) / valid if valid > 0 else 0.0
    partial = b / valid if valid > 0 else 0.0
    parametric = (b + c) / valid if valid > 0 else 0.0
    pp = a / (a + b) if (a + b) > 0 else 0.0
    return {"PAR": par, "Partial": partial, "PR": parametric, "PP": pp, "valid": valid}


def print_stats(dist: dict, total: int, model_label: str):
    m = compute_metrics(dist)
    valid = m["valid"]

    def pct(n):
        return f"{n / valid * 100:.1f}%" if valid > 0 else "N/A"

    print(f"\n{'=' * 55}")
    print(f"  Attribution Report -- {model_label}")
    print(f"{'=' * 55}")
    print(f"  Total processed : {total}")
    print(f"  Valid judgements: {valid}")
    print(f"  Errors          : {dist.get('error', 0)}")
    print()
    print(f"  Label distribution:")
    print(
        f"    [A] Fully grounded in passage          : {dist.get('A', 0):4d}  ({pct(dist.get('A', 0))})"
    )
    print(
        f"    [B] Passage + parametric               : {dist.get('B', 0):4d}  ({pct(dist.get('B', 0))})"
    )
    print(
        f"    [C] Parametric only                    : {dist.get('C', 0):4d}  ({pct(dist.get('C', 0))})"
    )
    print()
    print(f"  Metrics:")
    print(f"    PAR  (Passage Attribution Rate, A+B)        : {m['PAR']:.3f}")
    print(f"    PR   (Parametric Rate, B+C)            : {m['PR']:.3f}")
    print(f"    PP   (Passage Precision, A/(A+B))      : {m['PP']:.3f}")
    print(f"{'=' * 55}\n")


def write_jsonl(path: str, records: list):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Written {len(records)} records -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--tag", type=str, default="out")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--model-label", default="Model")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    import os

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    llm = LLM(model=args.model, dtype="half", max_model_len=16384)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.test:
        records = records[:20]

    total = len(records)
    print(f"Loaded {total} records.")

    prompts = []
    meta = []

    for rec in records:
        try:
            prompt_text = rec["prompt"]
            # ← 关键修复：从 raw_output 解析 sample，而不是 rec["sample"]
            sample = json.loads(rec["raw_output"])

            question = sample["question"]
            options = sample["options"]
            answer_letter = sample["answer"].strip().upper()[0]
            answer_text = get_answer_text(options, answer_letter)
            passage = extract_passage(prompt_text)

            chat_prompt = build_chat_prompt(
                tokenizer, passage, question, answer_letter, answer_text
            )

            if len(prompts) == 0:
                print("\n===== PROMPT EXAMPLE =====")
                print(chat_prompt)
                print("===== END PROMPT =====\n")

            prompts.append(chat_prompt)
            meta.append({"rec": rec, "sample": sample, "error": None})

        except Exception as e:
            prompts.append(None)
            meta.append({"rec": rec, "sample": None, "error": str(e)})

    valid_indices = [i for i, p in enumerate(prompts) if p is not None]
    valid_prompts = [prompts[i] for i in valid_indices]

    print(f"Running inference on {len(valid_prompts)} prompts...")
    outputs = llm.generate(valid_prompts, sampling_params)
    output_map = {idx: out.outputs[0].text for idx, out in zip(valid_indices, outputs)}

    distribution = {"A": 0, "B": 0, "C": 0, "error": 0}

    a_clean, a_raw = [], []
    b_clean, b_raw = [], []

    with open(args.output, "w", encoding="utf-8") as fout:
        for i, m in enumerate(meta):
            rec = m["rec"]
            if m["error"]:
                rec["attribution"] = {"label": "error", "justification": m["error"]}
                distribution["error"] += 1
            else:
                raw_output = output_map.get(i, "")
                try:
                    result = parse_judge_output(raw_output)
                    label = result.get("label", "").strip().upper()
                    if label not in ("A", "B", "C"):
                        raise ValueError(f"Unexpected label: {label}")
                    rec["attribution"] = {
                        "label": label,
                        "justification": result.get("justification", ""),
                    }
                    distribution[label] += 1

                    sample = m["sample"]
                    clean_record = {
                        "question": sample.get("question", ""),
                        "options": sample.get("options", []),
                        "answer": sample.get("answer", ""),
                    }
                    raw_record = {
                        "prompt": rec.get("prompt", ""),
                        "raw_output": rec.get("raw_output", ""),
                    }

                    if label == "A":
                        a_clean.append(clean_record)
                        a_raw.append(raw_record)
                    elif label == "B":
                        b_clean.append(clean_record)
                        b_raw.append(raw_record)

                    if args.test:
                        print(
                            f"[{i + 1:02d}] label={label} | {result.get('justification', '')}"
                        )

                except Exception as e:
                    rec["attribution"] = {
                        "label": "error",
                        "justification": f"{e} | raw: {raw_output[:100]}",
                    }
                    distribution["error"] += 1
                    if args.test:
                        print(f"[line {i + 1}] ERROR: {e} | raw: {raw_output[:100]}")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tag = args.tag
    outdir = args.output_dir
    print("\nWriting split outputs:")
    write_jsonl(f"{outdir}/clean_samples_{tag}_A.jsonl", a_clean)
    write_jsonl(f"{outdir}/clean_raw_generations_{tag}_A.jsonl", a_raw)
    write_jsonl(f"{outdir}/clean_samples_{tag}_B.jsonl", b_clean)
    write_jsonl(f"{outdir}/clean_raw_generations_{tag}_B.jsonl", b_raw)

    print_stats(distribution, total, args.model_label)
    print(f"Full attributed output written to: {args.output}")


if __name__ == "__main__":
    main()
