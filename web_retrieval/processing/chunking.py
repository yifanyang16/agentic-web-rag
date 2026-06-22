# processing/chunker.py

import re
from typing import List

_ABBREV = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Fig|No|Vol|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\."
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _mask_abbreviations(text: str) -> tuple[str, dict]:
    """Replace known abbreviation periods with a placeholder to protect them."""
    placeholders = {}

    def replacer(m):
        token = m.group(0)
        key = f"__ABBREV{len(placeholders)}__"
        placeholders[key] = token
        return key

    masked = _ABBREV.sub(replacer, text)
    return masked, placeholders


def _restore_abbreviations(sentences: List[str], placeholders: dict) -> List[str]:
    restored = []
    for s in sentences:
        for key, val in placeholders.items():
            s = s.replace(key, val)
        restored.append(s)
    return restored


def clean_text(text: str) -> str:
    """
    Sentence-level noise filter.
    Removes ads, boilerplate, CTA fragments, and near-duplicate sentences.
    """
    text = text.replace("\\n", "\n")

    masked, placeholders = _mask_abbreviations(text.strip())
    raw_sentences = SENTENCE_SPLIT_RE.split(masked)
    sentences = _restore_abbreviations(
        [s.strip() for s in raw_sentences if s.strip()],
        placeholders,
    )

    cleaned = []
    seen = set()

    for s in sentences:
        s_lower = s.lower()

        if s_lower in {"advertisement", "policy"}:
            continue

        if any(
            k in s_lower
            for k in [
                "non-profit academic medical center",
                "we do not endorse",
                "advertising on our site",
                "health essentials emails",
                "our mission",
                "editorial process",
                "health library",
            ]
        ):
            continue

        if any(
            k in s_lower
            for k in [
                "learn more",
                "subscribe",
                "sign up",
                "email",
                "contact us",
                "for expert guidance",
                "help you manage",
            ]
        ):
            continue

        if s.endswith(":"):
            continue

        if len(s.split()) < 8:
            continue

        if s not in seen:
            seen.add(s)
            cleaned.append(s)

    return " ".join(cleaned)


def natural_chunk_text(
    text: str,
    max_chars: int = 800,
    target_chars: int = 600,
    min_sentences: int = 3,
    max_sentences: int = 8,
) -> List[str]:
    """
    Sentence-boundary chunking with a preferred target size and a hard cap.
    clean_text 只在这里调用一次。
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    masked, placeholders = _mask_abbreviations(cleaned)
    raw_sentences = SENTENCE_SPLIT_RE.split(masked)
    sentences = _restore_abbreviations(
        [s.strip() for s in raw_sentences if s.strip()],
        placeholders,
    )

    chunks: List[str] = []
    current: List[str] = []

    for s in sentences:
        current.append(s)
        joined = " ".join(current)

        if len(joined) >= target_chars and len(current) >= min_sentences:
            chunks.append(joined)
            current = []
            continue

        if len(joined) >= max_chars or len(current) >= max_sentences:
            chunks.append(joined)
            current = []

    if current:
        tail = " ".join(current)
        if chunks and len(tail) < 200:
            chunks[-1] += " " + tail
        else:
            chunks.append(tail)

    return chunks
