import numpy as np

from paper_explorer.data.models import Paper
from paper_explorer.embeddings.embedding_model import EmbeddingModel


class FakeSentenceTransformer:
    def __init__(self, dim: int = 4):
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self,
               texts,
               show_progress_bar = False,
               convert_to_numpy = True,
               normalize_embeddings = True):
        vectors = np.array([[float(len(t) % 7 + 1)] * self.dim for t in texts], dtype = np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis = 1, keepdims = True)
            vectors = vectors / norms
        return vectors


def test_embed_texts_returns_normalized_vectors():
    model = EmbeddingModel(model = FakeSentenceTransformer(dim = 4))
    vectors = model.embed_texts(["hello world", "a different, longer sentence here"])
    assert vectors.shape == (2, 4)
    norms = np.linalg.norm(vectors, axis = 1)
    assert np.allclose(norms, 1.0, atol = 1e-5)


def test_embed_papers_sets_embedding_field():
    model = EmbeddingModel(model = FakeSentenceTransformer(dim = 4))
    papers = [Paper(paper_id = "1", title = "Title", abstract = "Abstract text")]
    model.embed_papers(papers)
    assert papers[0].embedding is not None
    assert len(papers[0].embedding) == 4
