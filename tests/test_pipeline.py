import numpy as np
import pytest

from paper_explorer.data.models import Paper
from paper_explorer.data.storage import PaperStore
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.search.bm25_index import BM25Index
from paper_explorer.search.hybrid import HybridSearcher
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.pipeline import run_search
from paper_explorer.search.searcher import PaperSearcher


class FakeSentenceTransformer:
    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        vectors = np.array(
            [[1.0, 0.0, 0.0, 0.0] if "graph" in t.lower() else [0.0, 1.0, 0.0, 0.0] for t in texts],
            dtype=np.float32,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


class FakeReranker:
    def rerank(self, query, candidates):
        for r in candidates:
            r.rerank_score = 1.0 if "boost" in r.paper.title.lower() else 0.0
        candidates.sort(key=lambda r: (-r.rerank_score, r.paper.paper_id))
        for rank, r in enumerate(candidates, start=1):
            r.rank = rank
        return candidates


def make_papers(n):
    return [
        Paper(
            paper_id=f"p{i}",
            title=f"Boost Paper {i}" if i % 5 == 0 else f"Paper {i}",
            abstract="graph content",
        )
        for i in range(n)
    ]


def make_two_group_papers():
    relevant = [
        Paper(
            paper_id=f"r{i}",
            title="Boost Paper r2" if i == 2 else f"Relevant Paper r{i}",
            abstract="graph content",
        )
        for i in range(10)
    ]
    irrelevant = [
        Paper(paper_id=f"i{i}", title=f"Irrelevant Paper i{i}", abstract="cooking content")
        for i in range(10)
    ]
    return relevant + irrelevant


def build_hybrid_searcher(papers):
    model = EmbeddingModel(model=FakeSentenceTransformer())
    model.embed_papers(papers)
    store = PaperStore("unused.json")
    store.add_many(papers)

    vector_index = VectorIndex()
    if papers:
        vector_index.build(
            [p.paper_id for p in papers], np.array([p.embedding for p in papers], dtype=np.float32)
        )
    searcher = PaperSearcher(store=store, index=vector_index, embedding_model=model)
    bm25_index = BM25Index()
    bm25_index.build(papers)
    return HybridSearcher(store=store, bm25_index=bm25_index, searcher=searcher)


def test_rerank_disabled_matches_direct_hybrid_search():
    papers = make_papers(5)
    hybrid = build_hybrid_searcher(papers)
    direct = hybrid.search("graph", mode="semantic", top_k=3)
    via_pipeline = run_search(hybrid, "graph", mode="semantic", top_k=3, rerank=False)
    assert [r.paper.paper_id for r in direct] == [r.paper.paper_id for r in via_pipeline]


def test_rerank_enabled_uses_candidate_pool_and_returns_top_k():
    papers = make_two_group_papers()
    hybrid = build_hybrid_searcher(papers)
    reranker = FakeReranker()

    results = run_search(
        hybrid, "graph", mode="semantic", top_k=3, candidate_k=10, rerank=True, reranker=reranker
    )
    assert len(results) == 3
    assert results[0].paper.paper_id == "r2"
    assert all(r.paper.paper_id.startswith("r") for r in results)


def test_rerank_requires_reranker_instance():
    papers = make_papers(3)
    hybrid = build_hybrid_searcher(papers)
    with pytest.raises(ValueError):
        run_search(hybrid, "graph", rerank=True, reranker=None)


def test_rerank_with_empty_store_returns_empty():
    hybrid = build_hybrid_searcher([])
    reranker = FakeReranker()
    results = run_search(hybrid, "graph", rerank=True, reranker=reranker, candidate_k=10, top_k=5)
    assert results == []


def test_candidate_k_smaller_than_top_k_still_returns_available_results():
    papers = make_papers(5)
    hybrid = build_hybrid_searcher(papers)
    reranker = FakeReranker()
    results = run_search(
        hybrid, "graph", mode="semantic", top_k=5, candidate_k=2, rerank=True, reranker=reranker
    )
    assert len(results) == 5


def test_top_k_larger_than_corpus_with_rerank_enabled():
    papers = make_papers(2)
    hybrid = build_hybrid_searcher(papers)
    reranker = FakeReranker()
    results = run_search(
        hybrid, "graph", mode="semantic", top_k=50, candidate_k=50, rerank=True, reranker=reranker
    )
    assert len(results) == 2
