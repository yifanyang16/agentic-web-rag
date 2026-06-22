"""
Agentic Retriever — multi-agent version

Hierarchy:
    direction (from DOMAIN_SEEDS)
        → topic         (TopicPlanningAgent)
            → subtopic  (SubtopicAgent)
                → query (QueryAgent)
                    → web_search → JudgeAgent → ControllerAgent

Dedup scoping:
    Topic:    cross-round, persistent pool
    Subtopic: within-round only, fresh pool each step
    Query:    within-round only, fresh pool each step

Early stopping:
    ControllerAgent monitors high-score document gain per step.
    Topic exhaustion naturally produces zero gain and is handled
    by ControllerAgent — no separate break condition.

Doc context:
    direction / topic / subtopic are bound onto each RetrievedDoc so
    the downstream pipeline (SelectorAgent) has full retrieval context
    without re-doing any LLM calls.
"""

import re
from typing import Dict, List, Tuple, Generator

import numpy as np
from difflib import SequenceMatcher
from llm_and_search import HelloAgentsLLM, RetrievedDoc, web_search
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Domain seeds
# ---------------------------------------------------------------------------

DOMAIN_SEEDS = {
    "bio": {
        "domain_name": "Biology",
        "output_file": "corpus_bio.jsonl",
        "directions": [
            "molecular biology",
            "genetics and genomics",
            "cell biology",
            "ecology and ecosystems",
            "evolutionary biology",
            "neuroscience",
            "microbiology",
            "developmental biology",
        ],
    },
    "med": {
        "domain_name": "Clinical Medicine",
        "output_file": "corpus_med.jsonl",
        "directions": [
            "cardiology",
            "oncology",
            "infectious disease",
            "neurology",
            "pharmacology",
            "immunology",
            "endocrinology",
            "pulmonology",
        ],
    },
}


# ---------------------------------------------------------------------------
# Embedding-based deduplication
# ---------------------------------------------------------------------------


class EmbeddingDedup:
    """
    Batch-encodes candidates in a single model.encode() call, then filters
    out items semantically similar to anything already in the pool or to
    each other (preserving input order).

    Instantiate once and reuse across calls for a persistent pool,
    or create a fresh instance each round for within-round-only dedup.
    """

    def __init__(self, model, threshold: float = 0.85) -> None:
        self.model = model
        self.threshold = threshold
        self._pool_vecs: List[np.ndarray] = []

    def filter(self, candidates: List[str]) -> List[str]:
        if not candidates:
            return []

        # single batch encode — O(1) model calls regardless of list length
        # shape: (N, D), L2-normalised → dot product == cosine similarity
        cand_vecs: np.ndarray = self.model.encode(candidates, normalize_embeddings=True)

        accepted_items: List[str] = []
        accepted_vecs: List[np.ndarray] = []

        for item, vec in zip(candidates, cand_vecs):
            if self._pool_vecs:
                pool_matrix = np.stack(self._pool_vecs)
                if (pool_matrix @ vec).max() > self.threshold:
                    continue
            if accepted_vecs:
                batch_matrix = np.stack(accepted_vecs)
                if (batch_matrix @ vec).max() > self.threshold:
                    continue
            accepted_items.append(item)
            accepted_vecs.append(vec)

        self._pool_vecs.extend(accepted_vecs)
        return accepted_items


# ---------------------------------------------------------------------------
# Shared text parsing helper
# ---------------------------------------------------------------------------


def parse_plain_lines(raw: str) -> List[str]:
    lines = raw.splitlines()
    items: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line[0] in "-•*":
            line = line[1:].strip()
        if len(line) > 2 and line[1] == "." and line[0].isdigit():
            line = line[2:].strip()
        lower = line.lower()
        if any(
            x in lower
            for x in (
                "certainly",
                "here are",
                "i'll",
                "let's",
                "description",
                "example output",
            )
        ):
            continue
        if len(line) > 160:
            continue
        items.append(line)

    seen: set = set()
    uniq: List[str] = []
    for q in items:
        if q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class TopicPlanningAgent:
    """
    Generates specific topics from a domain direction.
    domain_name drives the prompt so it is never hardcoded to one field.
    Dedup pool injected by caller — pass the persistent pool each round
    for cross-round dedup.
    """

    def __init__(self, llm: HelloAgentsLLM) -> None:
        self.llm = llm

    def plan_topics(
        self,
        domain_name: str,
        domain_directions: List[str],
        dedup: EmbeddingDedup,
    ) -> List[Tuple[str, str]]:
        """Returns list of (direction, topic) pairs that passed dedup."""
        results: List[Tuple[str, str]] = []

        for direction in domain_directions:
            raw = self.llm.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Generate 4 distinct {domain_name} topics for the"
                            f" following research direction:\n\n{direction}\n\n"
                            "Rules:\n"
                            "- each topic should be a specific concept, mechanism,"
                            " or process within the direction\n"
                            "- specific to have a dedicated Wikipedia section or university"
                            " course lecture, but NOT so narrow it only appears in research papers\n"
                            "- no paraphrasing between topics\n"
                            "- one topic per line, no bullets or numbering"
                            "Examples of good topics:\n"
                            "- DNA replication\n"
                            "- How CRISPR works\n"
                            "- Cell division and the cell cycle\n"
                            "- Immune response to viral infection\n\n"
                            "Examples of bad topics (too narrow, paper-style):\n"
                            "- CRISPR-Cas9-mediated epigenetic editing in stem cell differentiation\n"
                            "- RNA splicing regulation in eukaryotic gene expression\n\n"
                        ),
                    }
                ],
                temperature=0.6,
            )
            candidates = parse_plain_lines(raw)
            for t in dedup.filter(candidates):
                results.append((direction, t))

        return results


class SubtopicAgent:
    """
    Generates focused subtopics under a topic.
    Subtopics should be concrete enough to produce precise search queries.
    Dedup pool injected by caller — pass a fresh pool each round.
    """

    def __init__(self, llm: HelloAgentsLLM) -> None:
        self.llm = llm

    def plan_subtopics(
        self,
        topic: str,
        dedup: EmbeddingDedup,
    ) -> List[str]:
        raw = self.llm.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Generate exactly 2 focused subtopics for research.\n\n"
                        f"Topic: {topic}\n\n"
                        "Rules:\n"
                        "- focus on core, well-established knowledge in the field, NOT niche or cutting-edge research topics\n"
                        "- focus on a specific concept, application, or real-world instance within the topic\n"
                        " Wikipedia article, university lecture, or institutional explanation\n"
                        "- aimed at university student level, not elementary"
                        " NOT cutting-edge research paper specifics\n"
                        "- no paraphrasing between subtopics\n"
                        "- one subtopic per line, no bullets or numbering"
                    ),
                }
            ],
            temperature=0.3,
        )
        candidates = parse_plain_lines(raw)
        return dedup.filter(candidates)[:2]


class QueryAgent:
    """
    Generates precise search queries for a subtopic.
    Dedup pool injected by caller — pass a fresh pool each round.
    """

    def __init__(self, llm: HelloAgentsLLM, queries_per_topic: int = 2) -> None:
        self.llm = llm
        self.queries_per_topic = queries_per_topic

    def propose_queries(
        self,
        subtopic: str,
        dedup: EmbeddingDedup,
    ) -> List[str]:
        raw = self.llm.chat(
            [
                {
                    "role": "user",
                    "content": (
                        f"Generate {self.queries_per_topic} search queries.\n\n"
                        f"Subtopic: {subtopic}\n\n"
                        "Rules:\n"
                        "- each query should target a different aspect of the subtopic\n"
                        "- use clear, precise terminology at university textbook level\n"
                        "- prefer queries that return encyclopedia or institutional pages\n"
                        "- append terms like 'explained', 'overview', 'how it works' where natural\n"
                        "- no paraphrasing between queries\n"
                        "- one query per line, no bullets or numbering"
                    ),
                }
            ],
            temperature=0.3,
        )
        candidates = parse_plain_lines(raw)
        return dedup.filter(candidates)[: self.queries_per_topic]


class JudgeAgent:
    """
    Relevance scoring for retrieved documents.

    Scores against the full four-level context: direction → topic → subtopic → query.
    Intentionally inclusive: any page covering a meaningful aspect of the topic
    counts as relevant. The query layer handles precision; judge filters off-topic pages.
    """

    def __init__(self, llm: HelloAgentsLLM) -> None:
        self.llm = llm

    def score_doc_relevance(
        self,
        doc: RetrievedDoc,
        direction: str,
        topic: str,
        subtopic: str,
        query: str,
    ) -> Tuple[float, str]:
        prompt = (
            "You are an expert information retrieval judge.\n\n"
            "Task:\n"
            "Evaluate whether the webpage is relevant to the research topic below.\n"
            "Be inclusive: if the page covers any meaningful aspect of the Topic,"
            " score it as relevant. Only give a low score if the page is clearly"
            " off-topic or unrelated to the field.\n\n"
            f"Research direction: {direction}\n"
            f"Topic: {topic}\n"
            f"Subtopic: {subtopic}\n"
            f"Query: {query}\n\n"
            f"Webpage:\nTitle: {doc.title}\nSnippet: {doc.snippet}\n\n"
            "Output format:\n"
            "Explanation: <one sentence reasoning>\n"
            "Score: <number between 0 and 1>"
        )

        try:
            resp = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        except Exception:
            return 0.0, "Explanation: LLM error"

        score, explanation = self._parse(resp)
        doc.relevance = score
        doc.relevance_explanation = explanation
        return score, explanation

    @staticmethod
    def _parse(text: str) -> Tuple[float, str]:
        score = 0.0
        explanation = "Explanation: missing"
        for line in text.splitlines():
            l = line.strip()
            if l.lower().startswith("explanation"):
                explanation = l
            if "score" in l.lower():
                m = re.search(r"([01](?:\.\d+)?)", l)
                if m:
                    try:
                        score = float(m.group(1))
                    except Exception:
                        score = 0.0
        return max(0.0, min(score, 1.0)), explanation


class ControllerAgent:
    """
    Monitors high-score document gain after each step.
    Stops early if gain is zero for `patience` consecutive steps.
    Single stopping mechanism — topic exhaustion feeds into this naturally.
    """

    def __init__(
        self,
        high_score_threshold: float = 0.6,
        patience: int = 2,
    ) -> None:
        self.high_score_threshold = high_score_threshold
        self.patience = patience
        self.consecutive_low_gain = 0

    def decide(
        self,
        prev_high: int,
        current_high: int,
        step: int,
        max_steps: int,
    ) -> Tuple[bool, str]:
        new_high = current_high - prev_high

        if new_high <= 0:
            self.consecutive_low_gain += 1
            reflection = (
                f"Step {step}: no new high-score documents. "
                f"Consecutive low-gain steps = {self.consecutive_low_gain}."
            )
        else:
            self.consecutive_low_gain = 0
            reflection = (
                f"Step {step}: +{new_high} high-score documents. "
                "Retrieval is still improving."
            )

        should_stop = False
        if self.consecutive_low_gain >= self.patience:
            should_stop = True
            reflection += (
                f" Stopping early: {self.consecutive_low_gain} consecutive"
                f" low-gain steps >= patience ({self.patience})."
            )
        elif step >= max_steps:
            should_stop = True
            reflection += " Reached maximum retrieval steps."

        return should_stop, reflection


# ---------------------------------------------------------------------------
# Retriever (orchestrator)
# ---------------------------------------------------------------------------


class AgenticRetriever:
    """
    Orchestrates the full retrieval pipeline for a single domain.

    After scoring, binds direction / topic / subtopic onto each accepted
    RetrievedDoc so the downstream pipeline has full retrieval context.
    """

    def __init__(
        self,
        llm: HelloAgentsLLM,
        embed_model,
        results_per_query: int = 4,
        max_steps: int = 1,
        relevance_threshold: float = 0.3,
        high_score_threshold: float = 0.6,
        controller_patience: int = 2,
        queries_per_topic: int = 2,
    ) -> None:
        self.embed_model = embed_model
        self.results_per_query = results_per_query
        self.max_steps = max_steps
        self.relevance_threshold = relevance_threshold
        self.high_score_threshold = high_score_threshold

        self.topic_agent = TopicPlanningAgent(llm)
        self.subtopic_agent = SubtopicAgent(llm)
        self.query_agent = QueryAgent(llm, queries_per_topic)
        self.judge_agent = JudgeAgent(llm)
        self.controller = ControllerAgent(high_score_threshold, controller_patience)

        self._topic_dedup = EmbeddingDedup(embed_model, threshold=0.83)

    @staticmethod
    def _norm_domain(url: str) -> str:
        try:
            return url.split("//", 1)[1].split("/", 1)[0].lower()
        except Exception:
            return url.lower()

    @staticmethod
    def _title_sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a[:100], b[:100]).ratio()

    def retrieve(
        self,
        domain_name: str,
        domain_directions: List[str],
    ) -> Generator[Tuple[str, List[RetrievedDoc]], None, None]:
        """
        Yields (direction, docs) one direction at a time.
        """
        for direction in domain_directions:
            collected_docs: Dict[str, RetrievedDoc] = {}

            def count_high_score() -> int:
                return sum(
                    1
                    for d in collected_docs.values()
                    if d.relevance >= self.high_score_threshold
                )

            for step in range(1, self.max_steps + 1):
                prev_high = count_high_score()

                subtopic_dedup = EmbeddingDedup(self.embed_model, threshold=0.85)
                query_dedup = EmbeddingDedup(self.embed_model, threshold=0.85)

                topics = self.topic_agent.plan_topics(
                    domain_name, [direction], self._topic_dedup
                )
                print(f"[Debug] topics: {topics}")
                if not topics:
                    print(f"[AgenticRetriever] Step {step}: no new topics this round.")

                for dir_, topic in topics:
                    subtopics = self.subtopic_agent.plan_subtopics(
                        topic, subtopic_dedup
                    )
                    print(f"[Debug] subtopics for '{topic}': {subtopics}")
                    for subtopic in subtopics:
                        queries = self.query_agent.propose_queries(
                            subtopic, query_dedup
                        )
                        print(f"[Debug] queries for '{subtopic}': {queries}")
                        for q in queries:
                            docs = web_search(q, self.results_per_query)
                            for d in docs:
                                if d.url in collected_docs:
                                    continue

                                score, _ = self.judge_agent.score_doc_relevance(
                                    d, dir_, topic, subtopic, q
                                )
                                if score < self.relevance_threshold:
                                    continue

                                domain = self._norm_domain(d.url)
                                title_key = d.title.strip().lower()

                                trustworthy = any(
                                    kw in domain
                                    for kw in (
                                        "wikipedia",
                                        "britannica",
                                        "wikiversity",
                                    )
                                )

                                blocklist = any(
                                    kw in domain
                                    for kw in (
                                        "reddit",
                                        "quora",
                                        "stackexchange",
                                        "stackoverflow",
                                        "medium.com",
                                        "substack",
                                        "blogspot",
                                        "wordpress",
                                        "wikihow",
                                        "ehow",
                                        "answers.com",
                                    )
                                )

                                trustworthy = trustworthy and not blocklist

                                duplicate = False
                                for existing_url, exist in collected_docs.items():
                                    same_domain = (
                                        self._norm_domain(existing_url) == domain
                                    )
                                    sim = self._title_sim(
                                        title_key, exist.title.strip().lower()
                                    )
                                    if (same_domain and sim > 0.88) or sim > 0.93:
                                        duplicate = True
                                        break

                                if duplicate and not trustworthy:
                                    continue

                                d.direction = dir_
                                d.topic = topic
                                d.subtopic = subtopic
                                collected_docs[d.url] = d

                current_high = count_high_score()
                should_stop, reflection = self.controller.decide(
                    prev_high, current_high, step, self.max_steps
                )
                print(f"[ControllerAgent] {reflection}")

                if should_stop:
                    break

            print(
                f"[AgenticRetriever] Direction '{direction}' done — {len(collected_docs)} docs collected."
            )
            yield direction, list(collected_docs.values())
