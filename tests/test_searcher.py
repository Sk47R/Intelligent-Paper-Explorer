import numpy as np

from paper_explorer.data.models import Paper
from paper_explorer.data.storage import PaperStore
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.searcher import PaperSearcher


class FakeSentenceTransformer:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, **kwargs):
        vectors = []
        for t in texts:
            vectors.append([1.0, 0.0, 0.0, 0.0] if "graph" in t.lower() else [0.0, 1.0, 0.0, 0.0])
        vectors = np.array(vectors, dtype = np.float32)
        norms = np.linalg.norm(vectors, axis = 1, keepdims = True)
        return vectors / norms


def test_searcher_returns_ranked_results(tmp_path):
    store = PaperStore(tmp_path / "papers.json")
    papers = [
        Paper(paper_id = "p1", title = "Graph Neural Networks", abstract = "about graphs"),
        Paper(paper_id = "p2", title = "Cooking Recipes", abstract = "about food"),
    ]
    model = EmbeddingModel(model = FakeSentenceTransformer(dim = 4))
    model.embed_papers(papers)
    store.add_many(papers)

    index = VectorIndex()
    index.build(
        [p.paper_id for p in papers], np.array([p.embedding for p in papers], dtype=np.float32)
    )

    searcher = PaperSearcher(store = store, index = index, embedding_model = model)
    results = searcher.search("graph learning", top_k = 2)

    assert results[0].paper.paper_id == "p1"
    assert results[0].rank == 1


def test_searcher_raises_on_empty_query(tmp_path):
    store = PaperStore(tmp_path / "papers.json")
    index = VectorIndex()
    model = EmbeddingModel(model = FakeSentenceTransformer(dim = 4))
    searcher = PaperSearcher(store = store, index = index, embedding_model = model)
    try:
        searcher.search("   ")
        assert False, "expected ValueError"
    except ValueError:
        pass
