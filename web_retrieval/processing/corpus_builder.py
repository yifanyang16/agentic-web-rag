import json
from typing import List, Dict


class CorpusBuilder:
    """
    Convert selected paragraphs into CRAFT-style corpus JSON records.

    Input: selector output
    [
      {
        "paragraph": "...",
        "score": 0.92,
        "explanation": "Explanation: ..."
      }
    ]

    Output (final JSONL):
    {
      "text": "...",
      "meta": {
         "url": "...",
         "topic": "...",
         "score": 0.92,
         "explanation": "...",
         "source": "web"
      }
    }
    """

    def build(self, url: str, topic: str, selected: List[Dict]) -> List[Dict]:
        """
        Convert selector output into corpus JSON records.
        """
        corpus = []

        if not selected:
            return corpus

        for item in selected:
            paragraph = item.get("paragraph", "")
            score = item.get("score", 0.0)
            explanation = item.get("explanation", "Explanation: None.")

            corpus.append(
                {
                    "text": paragraph,
                    "meta": {
                        "url": url,
                        "topic": topic,
                        "score": score,
                        "explanation": explanation,
                        "source": "web",
                    },
                }
            )

        return corpus

    def save_jsonl(self, records: List[Dict], path: str):
        """
        Save final corpus to JSONL format.
        """
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
