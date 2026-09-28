import numpy as np

from paper_explorer.search.index import VectorIndex


def test_build_and_search_returns_best_match_first():
    ids = ["a", "b", "c"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype = np.float32)
    index = VectorIndex()
    index.build(ids, embeddings)
    results = index.search(np.array([1.0, 0.0], dtype = np.float32), top_k = 2)
    result_ids = [pid for pid, _ in results]
    assert result_ids[0] == "a"
    assert "b" not in result_ids


def test_save_and_load_roundtrip(tmp_path):
    ids = ["a", "b"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype = np.float32)
    index = VectorIndex()
    index.build(ids, embeddings)

    index_path = tmp_path / "index.faiss"
    id_map_path = tmp_path / "id_map.json"
    index.save(index_path, id_map_path)

    reloaded = VectorIndex.load(index_path, id_map_path)
    results = reloaded.search(np.array([1.0, 0.0], dtype = np.float32), top_k = 1)
    assert results[0][0] == "a"


def test_empty_index_search_returns_empty_list():
    index = VectorIndex()
    results = index.search(np.array([1.0, 0.0], dtype = np.float32), top_k = 5)
    assert results == []
