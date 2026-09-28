from __future__ import annotations

import logging

from paper_explorer.data.storage import PaperStore
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.results import SearchResult

logger = logging.getLogger(__name__)

class PaperSearcher:
    def __init__(
        self,
        store: PaperStore,
        index: VectorIndex,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.store = store
        self.index = index
        self.embedding_model = embedding_model or EmbeddingModel()

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if not query or not query.strip():
            raise ValueError("Search query must not be empty")
        if not self.index.is_built:
            raise RuntimeError("Vector index is empty. Run `ingest` before searching.")

        query_vector = self.embedding_model.embed_texts([query])[0]
        hits = self.index.search(query_vector, top_k = top_k)

        results = []
        for rank, (paper_id, score) in enumerate(hits, start = 1):
            paper = self.store.get(paper_id)
            if paper is None:
                logger.warning(
                    "Paper %s found in index but missing from store; skipping", paper_id
                )
                continue
            results.append(SearchResult(rank = rank, paper = paper, score = score))
        return results
