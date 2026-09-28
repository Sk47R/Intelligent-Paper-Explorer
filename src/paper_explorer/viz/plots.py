from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
from sklearn.decomposition import PCA

from paper_explorer.data.models import Paper

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_reranking_impact(results: list, output_path) -> Path:
    from paper_explorer.search.display import normalize_min_max, sigmoid

    scored = [r for r in results if r.rerank_score is not None]
    if not scored:
        raise ValueError("No reranked results to plot (rerank_score is None on all results)")

    retrieval_raw = [r.score for r in scored]
    retrieval_norm = normalize_min_max(retrieval_raw)
    rerank_norm = [sigmoid(r.rerank_score) for r in scored]

    retrieval_order = sorted(range(len(scored)), key=lambda i: -retrieval_raw[i])
    retrieval_rank_of = {orig_idx: rank for rank, orig_idx in enumerate(retrieval_order, start=1)}
    movement = [abs(retrieval_rank_of[i] - scored[i].rank) for i in range(len(scored))]

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        retrieval_norm,
        rerank_norm,
        c=movement,
        cmap="viridis",
        s=70,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
        label="y = x (rerank agrees with retrieval)",
    )
    ax.set_xlabel("First-stage retrieval score (normalized, 0-1)")
    ax.set_ylabel("Cross-encoder rerank score (sigmoid, 0-1)")
    ax.set_title("Reranking impact: retrieval score vs. rerank score")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("|rank change| (retrieval-only rank vs. final rank)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_mode_overlap(semantic_results, keyword_results, hybrid_results, output_path) -> Path:

    sem_ids = {r.paper.paper_id for r in semantic_results}
    key_ids = {r.paper.paper_id for r in keyword_results}
    hyb_ids = {r.paper.paper_id for r in hybrid_results}

    only_sem = sem_ids - key_ids - hyb_ids
    only_key = key_ids - sem_ids - hyb_ids
    only_hyb = hyb_ids - sem_ids - key_ids
    sem_key = (sem_ids & key_ids) - hyb_ids
    sem_hyb = (sem_ids & hyb_ids) - key_ids
    key_hyb = (key_ids & hyb_ids) - sem_ids
    all_three = sem_ids & key_ids & hyb_ids

    labels = [
        "Semantic\nonly",
        "Keyword\nonly",
        "Hybrid\nonly",
        "Semantic\n& Keyword",
        "Semantic\n& Hybrid",
        "Keyword\n& Hybrid",
        "All\nthree",
    ]
    counts = [
        len(only_sem),
        len(only_key),
        len(only_hyb),
        len(sem_key),
        len(sem_hyb),
        len(key_hyb),
        len(all_three),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, counts, color="#3366cc")
    ax.set_ylabel("Number of papers")
    ax.set_title("Overlap of top-K results across search modes (same query)")
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(count),
                ha="center",
                va="bottom",
            )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

def plot_papers_per_year(papers: list[Paper], output_path: str | Path) -> Path:
    years = [p.published[:4] for p in papers if p.published]
    counts = Counter(years)
    ordered = sorted(counts.items())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([year for year, _ in ordered], [count for _, count in ordered], color="#3366cc")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.set_title("Papers per year")
    plt.xticks(rotation=45)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_embedding_space(
    papers: list[Paper], output_path: str | Path, labels: list[int] | None = None
) -> Path:
    embedded = [p for p in papers if p.embedding is not None]
    matrix = np.array([p.embedding for p in embedded], dtype=np.float32)
    coords = PCA(n_components=2, random_state=42).fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
    ax.set_title("Paper embedding space (PCA projection)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    if labels is not None:
        legend = ax.legend(*scatter.legend_elements(), title="Topic", loc="best")
        ax.add_artist(legend)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
