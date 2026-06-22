import os
from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from serpapi import GoogleSearch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


@dataclass
class RetrievedDoc:
    url: str
    title: str
    snippet: str

    relevance: float = 0.0
    relevance_explanation: str = ""

    direction: str = ""
    topic: str = ""
    subtopic: str = ""


VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B")


class HelloAgentsLLM:
    def __init__(self, model: Optional[str] = None) -> None:
        self.model_name = model or VLLM_MODEL
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.90,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True, trust_remote_code=True
        )

    def chat(self, messages: list, temperature: float = 0.2) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=1024,
        )
        outputs = self.llm.generate([prompt], sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()


_EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class LocalEmbeddingModel:
    def __init__(self, model_name: str = _EMBED_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], normalize_embeddings: bool = True):
        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )


EXCLUDE_SITES = " ".join(
    [
        "-site:pmc.ncbi.nlm.nih.gov",
        "-site:sciencedirect.com",
        "-site:nature.com",
        "-site:pnas.org",
        "-site:academic.oup.com",
        "-site:cell.com",
        "-site:onlinelibrary.wiley.com",
        "-site:journals.sagepub.com",
        "-site:genesdev.cshlp.org",
        "-site:mdpi.com",
        "-site:portlandpress.com",
        "-site:journals.plos.org",
        "-site:annualreviews.org",
        "-site:elifesciences.org",
    ]
)


def web_search(query: str, num_results: int = 8) -> List[RetrievedDoc]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY is not configured.")

    params = {
        "engine": "google",
        "q": f"{query} {EXCLUDE_SITES}",
        "api_key": api_key,
        "gl": "us",
        "hl": "en",
        "num": num_results,
    }

    results = GoogleSearch(params).get_dict()

    docs: List[RetrievedDoc] = []
    for item in results.get("organic_results", [])[:num_results]:
        if not item.get("link"):
            continue
        docs.append(
            RetrievedDoc(
                title=item.get("title", "") or "",
                snippet=item.get("snippet", "") or "",
                url=item.get("link", "") or "",
            )
        )
    return docs
