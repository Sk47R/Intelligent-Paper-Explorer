from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorIndex:
    def __init__(self, dim: int | None = None):
        self.dim = dim
        self._index: faiss.IndexFlatIP | None = None
        self.id_map: list[str] = []

    @property
    def is_built(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def build(self, paper_ids: list[str], embeddings: np.ndarray) -> None:
        if len(paper_ids) != embeddings.shape[0]:
            raise ValueError("paper_ids and embeddings must have the same length")
        self.dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.id_map = list(paper_ids)
        logger.info("Built FAISS index with %d vectors (dim=%d)", len(paper_ids), self.dim)

    def __len__(self) -> int:
        return 0 if self._index is None else self._index.ntotal


    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.is_built:
            return []

        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        top_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.id_map[idx], float(score)))

        return results

    def save(self, index_path: str | Path, id_map_path: str | Path) -> None:
        if self._index is None:
            raise ValueError("Cannot save an empty index; call build() first")

        index_path = Path(index_path)
        id_map_path = Path(id_map_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        id_map_path.write_text(json.dumps(self.id_map), encoding="utf-8")
        logger.info("Saved FAISS index to %s and id map to %s", index_path, id_map_path)

    @classmethod
    def load(cls, index_path: str | Path, id_map_path: str | Path) -> VectorIndex:
        index_path = Path(index_path)
        id_map_path = Path(id_map_path)
        faiss_index = faiss.read_index(str(index_path))
        id_map = json.loads(id_map_path.read_text(encoding="utf-8"))

        instance = cls(dim=faiss_index.d)
        instance._index = faiss_index
        instance.id_map = id_map
        return instance
