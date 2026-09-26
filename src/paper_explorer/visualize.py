from __future__ import annotations
from rich.console import Console
from paper_explorer.data.repository import PaperRepository
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.search.bm25_index import BM25Index
from paper_explorer.search.hybrid import HybridSearcher
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.pipeline import run_search
from paper_explorer.search.reranker import CrossEncoderReranker, RerankerLoadError
from paper_explorer.search.searcher import PaperSearcher
from paper_explorer.viz.plots import plot_mode_overlap, plot_reranking_impact

import logging
logger = logging.getLogger(__name__)
console = Console()


def run_visualize(
    db_path,
    index_path,
    id_map_path,
    query: str,
    output_dir,
    top_k: int = 10,
    candidate_k: int = 50,
    alpha: float = 0.5,
) -> None:
    repository = PaperRepository(db_path)
    if len(repository) == 0:
        console.print("[red]No papers in database. Run `ingest` first.[/red]")
        raise SystemExit(1)

    try:
        vector_index = VectorIndex.load(index_path, id_map_path)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        console.print(f"[red]No FAISS index found at {index_path}. Run `ingest` first.[/red]")
        raise SystemExit(1) from exc

    embedding_model = EmbeddingModel()
    searcher = PaperSearcher(store=repository, index=vector_index, embedding_model=embedding_model)

    bm25_index = BM25Index()
    bm25_index.build(repository.all())

    hybrid_searcher = HybridSearcher(store=repository, bm25_index=bm25_index, searcher=searcher)

    console.print(f"[dim]Running semantic/keyword/hybrid search for {query!r} ...[/dim]")
    semantic_results = run_search(hybrid_searcher, query, mode="semantic", top_k=top_k)
    keyword_results = run_search(hybrid_searcher, query, mode="keyword", top_k=top_k)
    hybrid_results = run_search(
        hybrid_searcher, query, mode="hybrid", alpha=alpha, candidate_k=candidate_k, top_k=top_k
    )

    console.print("[dim]Loading reranker...[/dim]")
    try:
        reranker = CrossEncoderReranker()
    except RerankerLoadError as exc:
        console.print(f"[red]Failed to load reranker:[/red] {exc}")
        raise SystemExit(1) from exc

    reranked_results = run_search(
        hybrid_searcher,
        query,
        mode="hybrid",
        alpha=alpha,
        candidate_k=candidate_k,
        top_k=top_k,
        rerank=True,
        reranker=reranker,
    )

    overlap_path = plot_mode_overlap(
        semantic_results, keyword_results, hybrid_results, f"{output_dir}/mode_overlap.png"
    )
    console.print(f"[green]Saved[/green] {overlap_path}")

    try:
        rerank_path = plot_reranking_impact(
            reranked_results, f"{output_dir}/reranking_impact.png"
        )
        console.print(f"[green]Saved[/green] {rerank_path}")
    except ValueError as exc:
        console.print(f"[yellow]Skipped reranking plot: {exc}[/yellow]")