from __future__ import annotations

import math

from paper_explorer.search.results import SearchResult


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def normalize_min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        fill = 1.0 if hi > 0 else 0.0
        return [fill for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def with_display_scores(results: list[SearchResult]) -> list[dict]:
    keyword_raw = [r.keyword_score for r in results if r.keyword_score is not None]
    keyword_norm_lookup = dict(
        zip(
            [r.paper.paper_id for r in results if r.keyword_score is not None],
            normalize_min_max(keyword_raw),
        )
    )

    out = []
    for r in results:
        breakdown = {}
        if r.semantic_score is not None:
            breakdown["semantic"] = clip01(r.semantic_score)
        if r.keyword_score is not None:
            breakdown["keyword"] = keyword_norm_lookup.get(r.paper.paper_id, 0.0)
        if r.rerank_score is not None:
            breakdown["rerank"] = sigmoid(r.rerank_score)

        if r.rerank_score is not None:
            display_score = breakdown["rerank"]
        elif r.hybrid_score is not None:
            display_score = clip01(r.hybrid_score)
        elif r.keyword_score is not None:
            display_score = breakdown["keyword"]
        else:
            display_score = clip01(r.score)

        out.append({"result": r, "display_score": display_score, "display_breakdown": breakdown})
    return out
