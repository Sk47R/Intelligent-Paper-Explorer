from __future__ import annotations
from paper_explorer.data.models import Paper
import numpy as np
import os
import warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            show_progress_bar = False,
            convert_to_numpy = True,
            normalize_embeddings = True,
        )
        return embeddings.astype(np.float32)

    def embed_papers(self, papers: list[Paper]) -> list[Paper]:
        if not papers:
            return papers

        texts = [paper.text_for_embedding for paper in papers]
        vectors = self.embed_texts(texts)

        for paper, vector in zip(papers, vectors):
            paper.embedding = vector.tolist()

        return papers
