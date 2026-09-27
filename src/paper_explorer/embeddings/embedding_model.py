from __future__ import annotations
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from paper_explorer.data.models import Paper

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model: SentenceTransformer | None = None):
        self.model_name = model_name
        if model is not None:
            self._model = model
        else:
            logger.info("Loading embedding model %r ...", model_name)
            self._model = SentenceTransformer(model_name)
        logger.info("Embedding model ready (dim=%d)", self.dimension)

    # @property
    # def dimension(self) -> int:
    #     return self._model.get_sentence_embedding_dimension()
    
    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            show_progress_bar = False,
            convert_to_numpy = True,
            normalize_embeddings = True,
        )
        return np.asarray(embeddings, dtype = np.float32)

    def embed_papers(self, papers: list[Paper]) -> list[Paper]:
        if not papers:
            return papers
        texts = [paper.text_for_embedding for paper in papers]
        vectors = self.embed_texts(texts)
        for paper, vector in zip(papers, vectors):
            paper.embedding = vector.tolist()
        return papers