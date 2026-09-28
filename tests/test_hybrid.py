import numpy as np
import pytest

from paper_explorer.data.models import Paper
from paper_explorer.data.storage import PaperStore
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.search.bm25_index import BM25Index
from paper_explorer.search.hybrid import HybridSearcher, min_max_normalize
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.searcher import PaperSearcher


class FakeSentenceTransformer:
    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, **kwargs):
        vectors = []
        for t in texts:
            vectors.append([1.0, 0.0, 0.0, 0.0] if "graph" in t.lower() else [0.0, 1.0, 0.0, 0.0])
        vectors = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


def make_paper(paper_id: str, title: str, abstract: str) -> Paper:
    return Paper(paper_id=paper_id, title=title, abstract=abstract)


def build_hybrid_searcher(papers: list[Paper]) -> HybridSearcher:
    model = EmbeddingModel(model=FakeSentenceTransformer())
    model.embed_papers(papers)

    store = PaperStore("unused-path-in-memory.json")
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



def test_min_max_normalize_basic_range():
    result = min_max_normalize({"a": 0.0, "b": 5.0, "c": 10.0})
    assert result["a"] == pytest.approx(0.0)
    assert result["b"] == pytest.approx(0.5)
    assert result["c"] == pytest.approx(1.0)


def test_min_max_normalize_empty_dict():
    assert min_max_normalize({}) == {}


def test_min_max_normalize_all_equal_positive():
    result = min_max_normalize({"a": 3.0, "b": 3.0})
    assert result == {"a": 1.0, "b": 1.0}


def test_min_max_normalize_all_equal_zero():
    result = min_max_normalize({"a": 0.0, "b": 0.0})
    assert result == {"a": 0.0, "b": 0.0}



def test_semantic_mode_matches_paper_searcher_directly():
    papers = [
        make_paper("p1", "Graph Neural Networks", "message passing on graphs"),
        make_paper("p2", "Cooking Recipes", "how to bake bread"),
    ]
    hybrid = build_hybrid_searcher(papers)
    results = hybrid.search("graph learning", mode="semantic", top_k=2)
    assert results[0].paper.paper_id == "p1"
    assert results[0].semantic_score is not None
    assert results[0].keyword_score is None


def test_keyword_mode_does_not_require_semantic_index():
    papers = [
        make_paper("p1", "Transformer Architectures", "attention mechanisms for NLP"),
        make_paper("p2", "Cooking Recipes", "how to bake bread"),
    ]
    model = EmbeddingModel(model=FakeSentenceTransformer())
    model.embed_papers(papers)
    store = PaperStore("unused-path.json")
    store.add_many(papers)
    bm25_index = BM25Index()
    bm25_index.build(papers)

    hybrid = HybridSearcher(store=store, bm25_index=bm25_index, searcher=None)
    results = hybrid.search("transformer NLP", mode="keyword", top_k=2)
    assert results[0].paper.paper_id == "p1"
    assert results[0].keyword_score is not None
    assert results[0].semantic_score is None


def test_hybrid_mode_requires_searcher():
    papers = [make_paper("p1", "A Paper", "an abstract")]
    bm25_index = BM25Index()
    bm25_index.build(papers)
    store = PaperStore("unused-path.json")
    store.add_many(papers)

    hybrid = HybridSearcher(store=store, bm25_index=bm25_index, searcher=None)
    with pytest.raises(RuntimeError):
        hybrid.search("a query", mode="hybrid", top_k=5)

def test_hybrid_alpha_zero_is_pure_keyword_ranking():
    papers = [
        make_paper("p1", "Graph Neural Networks", "keyword match graph"),
        make_paper("p2", "Cooking Recipes", "no relevant keyword here at all"),
        make_paper("p3", "Astronomy Basics", "stars and planets and telescopes"),
    ]
    hybrid = build_hybrid_searcher(papers)
    results = hybrid.search("graph", mode="hybrid", alpha=0.0, top_k=3, candidate_k=10)
    assert results[0].paper.paper_id == "p1"
    assert results[0].hybrid_score == pytest.approx(results[0].keyword_score)

def test_hybrid_alpha_one_is_pure_semantic_ranking():
    papers = [
        make_paper("p1", "Graph Neural Networks", "message passing"),
        make_paper("p2", "Unrelated Cooking Topic", "bread and butter"),
    ]
    hybrid = build_hybrid_searcher(papers)
    results = hybrid.search("graph learning", mode="hybrid", alpha=1.0, top_k=2, candidate_k=10)
    assert results[0].paper.paper_id == "p1"
    assert results[0].hybrid_score == pytest.approx(results[0].semantic_score)


def test_hybrid_merges_and_deduplicates_candidates_present_in_both():
    papers = [
        make_paper("p1", "Graph Neural Networks", "graph graph graph message passing"),
        make_paper("p2", "Cooking Recipes", "bread and butter"),
    ]
    hybrid = build_hybrid_searcher(papers)
    results = hybrid.search("graph", mode="hybrid", top_k=10, candidate_k=10)
    ids = [r.paper.paper_id for r in results]
    assert ids.count("p1") == 1


def test_hybrid_top_k_larger_than_corpus_returns_all_available():
    papers = [make_paper("p1", "Only Paper", "the only one here")]
    hybrid = build_hybrid_searcher(papers)
    results = hybrid.search("only paper", mode="hybrid", top_k=50, candidate_k=50)
    assert len(results) == 1


def test_hybrid_empty_store_returns_empty_results():
    hybrid = build_hybrid_searcher([])
    results = hybrid.search("anything", mode="hybrid", top_k=5, candidate_k=5)
    assert results == []


def test_invalid_mode_raises_value_error():
    papers = [make_paper("p1", "A Paper", "an abstract")]
    hybrid = build_hybrid_searcher(papers)
    with pytest.raises(ValueError):
        hybrid.search("query", mode="not-a-real-mode")


def test_alpha_out_of_range_raises_value_error():
    papers = [make_paper("p1", "A Paper", "an abstract")]
    hybrid = build_hybrid_searcher(papers)
    with pytest.raises(ValueError):
        hybrid.search("query", mode="hybrid", alpha=1.5)


def test_empty_query_raises_value_error():
    papers = [make_paper("p1", "A Paper", "an abstract")]
    hybrid = build_hybrid_searcher(papers)
    with pytest.raises(ValueError):
        hybrid.search("   ", mode="keyword")
