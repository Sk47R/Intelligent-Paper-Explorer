from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from paper_explorer.data.models import Paper

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self.paper_ids: list[str] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None and len(self.paper_ids) > 0

    def build(self, papers: list[Paper]) -> None:
        self.paper_ids = [p.paper_id for p in papers]
        if not self.paper_ids:
            self._bm25 = None
            return
        corpus = [tokenize(p.text_for_embedding) for p in papers]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.is_built:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(query_tokens))
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.paper_ids[i], float(scores[i])) for i in top_indices]
