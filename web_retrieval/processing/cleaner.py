"""
Advanced cleaning utilities for webpage text extraction.

This module integrates:
- HTML/JS/CSS noise removal
- navigation/footer junk removal
- UI-instruction filtering (click/upload/etc)
- extremely short / title-only / caption filtering
- paragraph segmentation
- table-like pattern filtering
- deduplication

Designed for preparing high-quality explanatory paragraphs.
"""

import re
from typing import List
from difflib import SequenceMatcher


def clean_text_basic(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"&[a-zA-Z]+;", " ", text)

    text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL)

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"{.*?}", " ", text, flags=re.DOTALL)

    text = re.sub(r"(Home|Login|Sign Up|Menu|Back to top)", " ", text, flags=re.I)

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_low_information_lines(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        s = line.strip()
        if len(s) < 4:
            continue

        if len(re.sub(r"[A-Za-z0-9]", "", s)) / len(s) > 0.6:
            continue

        if re.search(r"(©|copyright|all rights reserved|privacy|cookies)", s, re.I):
            continue

        cleaned.append(s)

    return "\n".join(cleaned)


UI_KEYWORDS = [
    "click",
    "upload",
    "paste",
    "submit",
    "server",
    "enter your",
    "input your",
    "proceed",
    "javascript",
    "cookie",
    "welcome",
]


def is_ui_instruction(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in UI_KEYWORDS)


def is_title_only(text: str) -> bool:
    return ("." not in text) and (len(text) < 80)


def is_figure_caption(text: str) -> bool:
    return text.strip().lower().startswith(("fig", "figure", "table", "supplemental"))


def split_into_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text)]
    cleaned = []

    for p in paras:
        if len(p) < 20:
            continue

        # if sum(ch.isdigit() for ch in p) / max(len(p), 1) > 0.3:
        # continue

        if is_ui_instruction(p):
            continue

        if is_title_only(p):
            continue
        if is_figure_caption(p):
            continue

        cleaned.append(p)

    return cleaned


def dedup_paragraphs(paragraphs: List[str]) -> List[str]:
    deduped = []
    for p in paragraphs:
        keep = True
        for q in deduped:
            sim = SequenceMatcher(None, p[:200], q[:200]).ratio()
            if sim > 0.88:
                keep = False
                break
        if keep:
            deduped.append(p)
    return deduped


def clean_and_segment(text: str) -> List[str]:
    """
    1. Basic HTML/JS cleaning
    2. Remove line-level garbage
    3. Split into paragraphs
    4. Apply UI+title+caption filtering
    5. Deduplicate

    Returns: list of high-quality paragraphs
    """
    t = clean_text_basic(text)
    t = remove_low_information_lines(t)
    paras = split_into_paragraphs(t)
    paras = dedup_paragraphs(paras)
    return paras
