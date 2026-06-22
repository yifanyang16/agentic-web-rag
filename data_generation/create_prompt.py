from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from transformers import AutoTokenizer

TaskName = Literal["bioqa_mc", "medqa_mc", "csqa_mc"]


class MetaInstructions:
    BIOQA_MC = """
        Role: University Biology Professor.
        Task: Create a JSONL multiple-choice question based on the text.

        Requirements:
        1. NO "reading comprehension" phrases (e.g., "According to the text", "The study shows", "Researchers found").
        2. Focus on SCIENTIFIC PRINCIPLES and LOGIC. Skip administrative details (dates, journals, funding).
        3. Start the question directly with the scientific fact. (e.g., "The primary mechanism by which Cyanobacteria oxygenated the atmosphere was...")
        4. Ensure distractors (wrong options) are biologically plausible, not obvious nonsense.
        Return ONLY a JSON object:
        {
          "question": "<question text>",
          "options": ["A. <option A>", "B. <option B>", "C. <option C>", "D. <option D>"],
          "answer": "A" | "B" | "C" | "D"
        }
    """

    MEDQA_MC = """
        Role: Medical licensing exam question writer (USMLE / MedMCQA style).
        Task: Create a multiple-choice question based on the text.

        Requirements:
        1. NO "reading comprehension" phrases (e.g., "According to the text", "The passage states").
        2. Write in the style of USMLE or MedMCQA exam questions.
        3. Distractors must be clinically plausible — common wrong answers a real student might choose,
           not obvious nonsense.
        4. The correct answer must be unambiguously supported by the source text.
        Return ONLY a JSON object:
        {
          "question": "<question text>",
          "options": ["A. <option A>", "B. <option B>", "C. <option C>", "D. <option D>"],
          "answer":  "A" | "B" | "C" | "D"
        }
    """

    CSQA_MC = """
        Role: Experienced writer and editor for a general knowledge encyclopedia.
        Task: Create a yes/no multiple-choice question based on the text.

        Requirements:
        1. NO "reading comprehension" phrases (e.g., "According to the text", "The text states").
        2. Generate exactly one question that is answerable with yes or no.
        3. The question should be about a practical fact or concept in the text.
        4. Answer the question with the correct letter label.
        Return ONLY a JSON object:
        {
          "question": "<question text>",
          "options": ["A. Yes", "B. No"],
          "answer": "A" | "B"
        }
    """


def build_prompt(
    task: TaskName,
    corpus_text: str,
) -> str:
    if task == "bioqa_mc":
        instruction = MetaInstructions.BIOQA_MC
    elif task == "medqa_mc":
        instruction = MetaInstructions.MEDQA_MC
    elif task == "csqa_mc":
        instruction = MetaInstructions.CSQA_MC
    else:
        raise ValueError(f"Unknown task: {task}")

    corpus_text = corpus_text.strip()
    parts: List[str] = [instruction, "\n\n"]

    if task in ("bioqa_mc", "medqa_mc", "csqa_mc"):
        parts.append("Passage:\n")
        parts.append(corpus_text)
        parts.append(
            "\n\nNow create ONE multiple-choice question for this passage. "
            "Return ONLY the JSON object.\n"
        )

    return "".join(parts)


def _extract_json_block(text: str):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    json_str = text[start : end + 1]
    try:
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Invalid JSON extracted: {e}\nExtracted:\n{json_str}")


def _normalize_answer_letter(s: str, options: list = None) -> str:
    s_stripped = s.strip()
    if s_stripped.upper() in ["A", "B", "C", "D"]:
        return s_stripped.upper()
    m = re.match(r"^([ABCD])\.", s_stripped.upper())
    if m:
        return m.group(1)
    if options:
        for i, opt in enumerate(options):
            opt_text = re.sub(r"^[ABCD]\.\s*", "", str(opt)).strip()
            if opt_text.lower() == s_stripped.lower():
                return ["A", "B", "C", "D"][i]
        for i, opt in enumerate(options):
            opt_text = re.sub(r"^[ABCD]\.\s*", "", str(opt)).strip()
            if opt_text.lower() in s_stripped.lower() and len(opt_text) > 10:
                return ["A", "B", "C", "D"][i]
    raise ValueError(f"Cannot extract answer letter from: {s!r}")


def parse_model_output(task: TaskName, raw_output: str) -> Dict[str, Any]:
    obj = _extract_json_block(raw_output)

    if task in ("bioqa_mc", "medqa_mc", "csqa_mc"):
        if not isinstance(obj, dict):
            raise ValueError("Parsed QA JSON is not an object.")
        keys_missing = [k for k in ("question", "options", "answer") if k not in obj]
        if keys_missing:
            raise ValueError(f"Missing keys in QA JSON: {keys_missing}")

        question = str(obj["question"]).strip()
        options_raw = obj["options"]
        answer_raw = str(obj["answer"]).strip()

        if not isinstance(options_raw, list) or len(options_raw) != 4:
            raise ValueError("Field 'options' must be a list of exactly 4 elements.")

        max_choices = ["A", "B", "C", "D"]
        options = []
        for i, opt in enumerate(options_raw):
            opt_text = re.sub(r"^[ABCD]\.\s*", "", str(opt)).strip()
            options.append(f"{max_choices[i]}. {opt_text}")

        answer = _normalize_answer_letter(answer_raw, options_raw)
        return {"question": question, "options": options, "answer": answer}

    else:
        raise ValueError(f"Unknown task: {task}")


@dataclass
class PromptRecord:
    index: int
    prompt: str


def filter_by_length(
    prompts: List[str],
    model_name: str,
    max_tokens: int = 4096,
) -> List[PromptRecord]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    kept: List[PromptRecord] = []
    for i, p in enumerate(prompts):
        tokenized = tokenizer(p, return_length=True, return_attention_mask=False)
        length = tokenized["length"]
        if isinstance(length, list):
            length = length[0]
        if length <= max_tokens:
            kept.append(PromptRecord(index=i, prompt=p))
    return kept
