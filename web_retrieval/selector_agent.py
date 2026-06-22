# agents/selector_agent.py  (relevant changes only)

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from llm_and_search import HelloAgentsLLM


def is_title_like(text: str) -> bool:
    t = text.strip()
    if len(t) < 15:
        return True
    if "." not in t and len(t) < 80:
        return True
    if t.lower().startswith(("welcome", "introduction", "overview")):
        return True
    return False


class SelectorAgent:
    def __init__(self, llm: HelloAgentsLLM, top_k: int = 5, max_workers: int = 8):
        self.llm = llm
        self.top_k = top_k
        self.max_workers = max_workers  # ← 新增

    def score_paragraph(self, paragraph: str, context: List[str]) -> Tuple[str, float]:
        # 与原来完全相同，不改
        context_preview = "\n".join(context[:3])
        prompt = f"""
You are an expert scientific information retrieval judge.

Task:
Evaluate whether the paragraph is a scientifically explanatory text
relevant to the given topic context.

Topic context:
{context_preview}

CRITICAL RULES:
- If the paragraph is a TITLE, HEADER, or WELCOME MESSAGE → Score = 0
- If it is UI/navigation/instructional text → Score = 0
- If it contains no scientific explanation → Score = 0

Otherwise:
- explain briefly why it is relevant
- assign a score between 0 and 1

Paragraph:
{paragraph}

Output format:
Explanation: ...
Score: X
"""
        try:
            resp = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        except Exception:
            return "Explanation: scoring failed.", 0.0

        return self._parse_explanation_and_score(resp)

    @staticmethod
    def _parse_explanation_and_score(text: str) -> Tuple[str, float]:
        m = re.search(r"Score:\s*([01](?:\.\d+)?)", text)
        score = float(m.group(1)) if m else 0.0
        m2 = re.search(r"([01](?:\.\d+)?)", text)
        if not m and m2:
            score = float(m2.group(1))
        score = max(0.0, min(score, 1.0))

        explanation = "Explanation: None."
        for line in text.splitlines():
            if line.lower().startswith("explanation:"):
                explanation = line.strip()
                break

        return explanation, score

    def select(
        self,
        paragraphs: List[str],
        context: List[str],
        threshold: float = 0.55,
    ) -> List[dict]:
        # 1. 预过滤（与原来相同）
        candidates = [
            p for p in paragraphs if not is_title_like(p) and len(p.strip()) >= 40
        ]

        # 2. 并发打分 ← 核心改动
        scored = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_para = {
                executor.submit(self.score_paragraph, p, context): p for p in candidates
            }
            for future in as_completed(future_to_para):
                p = future_to_para[future]
                try:
                    explanation, score = future.result()
                except Exception:
                    explanation, score = "Explanation: scoring failed.", 0.0

                scored.append(
                    {
                        "paragraph": p,
                        "score": score,
                        "explanation": explanation,
                    }
                )

        # 3. 排序 + 过滤（与原来相同）
        scored.sort(key=lambda x: x["score"], reverse=True)
        filtered = [x for x in scored if x["score"] >= threshold]
        return filtered[: self.top_k]
