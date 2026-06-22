"""
pipeline.py

Full retrieval → extraction → selection → corpus pipeline.

Usage:
    python pipeline.py --task bio
    python pipeline.py --task med
    python pipeline.py --task cs
"""

import argparse
import json

from agentic_retrieval_med import DOMAIN_SEEDS, AgenticRetriever
from extractor import WebpageExtractor
from selector_agent import SelectorAgent
from corpus_builder import CorpusBuilder
from llm_and_search import HelloAgentsLLM, LocalEmbeddingModel
from chunking import natural_chunk_text


def run_pipeline(
    task: str = "med",
    max_steps: int = 3,
    queries_per_topic: int = 2,
    results_per_query: int = 5,
    relevance_threshold: float = 0.3,
    selector_threshold: float = 0.55,
    selector_top_k: int = 5,
):
    config = DOMAIN_SEEDS[task]
    domain_name = config["domain_name"]
    directions = config["directions"]
    output_file = config["output_file"]

    print(f"[Pipeline] Starting task '{task}' — domain: {domain_name}")

    # --- shared components ---
    llm = HelloAgentsLLM()
    embed_model = LocalEmbeddingModel()
    extractor = WebpageExtractor()
    selector = SelectorAgent(llm, top_k=selector_top_k)
    builder = CorpusBuilder()

    # --- retrieval ---
    retriever = AgenticRetriever(
        llm=llm,
        embed_model=embed_model,
        max_steps=max_steps,
        queries_per_topic=queries_per_topic,
        results_per_query=results_per_query,
        relevance_threshold=relevance_threshold,
    )

    total_written = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for direction, docs in retriever.retrieve(domain_name, directions):
            print(
                f"\n[Pipeline] Processing direction '{direction}' — {len(docs)} docs retrieved."
            )

            direction_written = 0

            for doc in docs:
                print(f"\n[Pipeline] Extracting: {doc.url}")

                paragraphs = extractor.extract(doc.url)
                if not paragraphs:
                    print(f"  → No paragraphs extracted, skipping.")
                    continue

                context = [doc.direction, doc.topic, doc.subtopic]

                selected = selector.select(
                    paragraphs,
                    context=context,
                    threshold=selector_threshold,
                )

                if not selected:
                    print(f"  → No paragraphs passed selection, skipping.")
                    continue

                print(f"  → {len(selected)} paragraphs selected.")

                chunked_selected = []
                for entry in selected:
                    chunks = natural_chunk_text(entry["paragraph"])
                    for chunk in chunks:
                        chunked_selected.append(
                            {
                                "paragraph": chunk,
                                "score": entry["score"],
                                "explanation": entry["explanation"],
                            }
                        )

                if not chunked_selected:
                    print(f"  → All paragraphs empty after chunking, skipping.")
                    continue

                print(f"  → {len(chunked_selected)} chunks after splitting.")

                corpus_entries = builder.build(doc.url, doc.topic, chunked_selected)
                for record in corpus_entries:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                direction_written += len(corpus_entries)
                total_written += len(corpus_entries)

            out_f.flush()
            print(
                f"[Pipeline] Direction '{direction}' done — {direction_written} records written. Total so far: {total_written}."
            )

    print(f"\n[Pipeline] Done. Saved {total_written} records → {output_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="bio", choices=list(DOMAIN_SEEDS.keys())
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--queries-per-topic", type=int, default=3)
    parser.add_argument("--results-per-query", type=int, default=5)
    parser.add_argument("--relevance-threshold", type=float, default=0.3)
    parser.add_argument("--selector-threshold", type=float, default=0.55)
    parser.add_argument("--selector-top-k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        task=args.task,
        max_steps=args.max_steps,
        queries_per_topic=args.queries_per_topic,
        results_per_query=args.results_per_query,
        relevance_threshold=args.relevance_threshold,
        selector_threshold=args.selector_threshold,
        selector_top_k=args.selector_top_k,
    )
