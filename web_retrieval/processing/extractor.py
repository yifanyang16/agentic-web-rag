"""
WebpageExtractor: Extract and clean webpage text into paragraphs.

Does NOT score, filter, or structure the output.
Selector and CorpusBuilder handle those stages separately.

Pipeline:
  1. fetch html
  2. extract raw main text
  3. clean & segment paragraphs (cleaner)
  4. return List[str] paragraphs
"""

import trafilatura
from typing import List

from cleaner import clean_and_segment


class WebpageExtractor:
    """
    Extract clean explanatory paragraphs from a webpage.
    """

    def fetch_html(self, url: str) -> str:
        downloaded = trafilatura.fetch_url(url)
        return downloaded or ""

    def extract_raw_text(self, html: str) -> str:
        # 第一轮：正常模式
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if extracted:
            return extracted

        extracted = trafilatura.extract(
            html,
            include_comments=True,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
        )
        return extracted or ""

    def extract(self, url: str) -> List[str]:
        """
        Full extraction pipeline WITHOUT selector:
            url → html → raw text → cleaned paragraphs (list[str])
        """
        html = self.fetch_html(url)
        if not html:
            return []

        raw_text = self.extract_raw_text(html)
        if not raw_text:
            return []

        paragraphs = clean_and_segment(raw_text)
        return paragraphs
