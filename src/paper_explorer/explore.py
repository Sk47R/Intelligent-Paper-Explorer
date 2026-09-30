from __future__ import annotations

import logging

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from paper_explorer.crawler.arxiv_client import ArxivAPIError, ArxivClient
from paper_explorer.data.repository import PaperRepository
from paper_explorer.embeddings.embedding_model import EmbeddingModel
from paper_explorer.ingestion.service import IngestionService
from paper_explorer.search.bm25_index import BM25Index
from paper_explorer.search.display import with_display_scores
from paper_explorer.search.hybrid import HybridSearcher
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.pipeline import run_search
from paper_explorer.search.reranker import CrossEncoderReranker, RerankerLoadError
from paper_explorer.search.searcher import PaperSearcher

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_COVERAGE_THRESHOLD = 0.35
AUTO_INGEST_MAX_RESULTS = 100


def _local_coverage_score(
    repository: PaperRepository,
    embedding_model: EmbeddingModel,
    index_path,
    id_map_path,
    query: str
) -> float:
    if len(repository) == 0:
        return -1.0
    try:
        vector_index = VectorIndex.load(index_path, id_map_path)
    except (FileNotFoundError, OSError, RuntimeError):
        return -1.0
    if not vector_index.is_built:
        return -1.0

    searcher = PaperSearcher(store=repository, index=vector_index, embedding_model=embedding_model)
    results = searcher.search(query, top_k=1)
    return results[0].score if results else -1.0


def _auto_ingest(client: ArxivClient,
                 repository: PaperRepository, embedding_model,
                 query: str, index_path, id_map_path) -> None:
    console.print(
        f"[yellow]Local library doesn't cover {query!r} yet...\n"
        f"ingesting {AUTO_INGEST_MAX_RESULTS} papers from arXiv...[/yellow]"
    )
    service = IngestionService(repository, client=client, embedding_model=embedding_model)
    try:
        summary = service.ingest(query, max_results=AUTO_INGEST_MAX_RESULTS)
    except ArxivAPIError as exc:
        console.print(f"[red]arXiv API error:[/red] {exc}")
        console.print("[yellow]Falling back to whatever is already in the local library.[/yellow]")
        return

    console.print(
        f"[green]Ingested[/green] (new={summary['new']}, changed={summary['changed']}, "
        f"unchanged={summary['unchanged']}, embedded={summary['embedded']})"
    )
    index = service.rebuild_index()
    if len(index) > 0:
        index.save(index_path, id_map_path)


def _print_results(display_rows: list[dict]) -> None:
    if not display_rows:
        console.print("[yellow]No results.[/yellow]")
        return
    print("\n\n")
    table = Table(title="Results (0.0 = weakest match, 1.0 = strongest match)")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Semantic", justify="right")
    table.add_column("Keyword", justify="right")
    table.add_column("arXiv ID")
    table.add_column("Title")

    for row in display_rows:
        r = row["result"]
        b = row["display_breakdown"]
        table.add_row(
            str(r.rank),
            f"{row['display_score']:.3f}",
            f"{b.get('semantic', float('nan')):.3f}" if "semantic" in b else "-",
            f"{b.get('keyword', float('nan')):.3f}" if "keyword" in b else "-",
            r.paper.paper_id,
            r.paper.title,
        )
    console.print(table)

    for row in display_rows:
        r = row["result"]
        console.print(
        f"\n[bold]{r.rank}. {r.paper.title}[/bold]  "
        f"(score: {row['display_score']:.3f})"
        )
        console.print(f"   Authors: {', '.join(r.paper.authors) or 'n/a'}")
        console.print(f"   {r.paper.abstract_url}")
        console.print(f"   {r.short_abstract()}")


def run_explore(
    db_path,
    index_path,
    id_map_path,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    top_k: int = 10,
    candidate_k: int = 50,
) -> None:
    repository = PaperRepository(db_path)
    client = ArxivClient()
    embedding_model = EmbeddingModel()

    console.print("[dim]Loading reranker (first run downloads a small model)...[/dim]")
    try:
        reranker = CrossEncoderReranker()
    except RerankerLoadError as exc:
        console.print(f"[red]Failed to load reranker:[/red] {exc}")
        return
    console.print("\n\n")
    console.print("[bold cyan]Paper Explorer: Interactive Search Engine[/bold cyan]")
    console.print("Type a query, or 'quit' to exit.\n")

    while True:
        query = Prompt.ask("Search query")
        if query.strip().lower() in ("quit", "exit", "q"):
            console.print("Goodbye!")
            break
        if not query.strip():
            continue

        coverage = _local_coverage_score(
            repository, embedding_model, index_path, id_map_path, query
        )
        if coverage < threshold:
            _auto_ingest(client, repository, embedding_model, query, index_path, id_map_path)

        if len(repository) == 0:
            console.print("[red]No papers available even after ingestion attempt.[/red]")
            continue

        bm25_index = BM25Index()
        bm25_index.build(repository.all())
        try:
            vector_index = VectorIndex.load(index_path, id_map_path)
        except (FileNotFoundError, OSError, RuntimeError):
            console.print("[red]No FAISS index available; cannot run semantic/hybrid search.[/red]")
            continue

        searcher = PaperSearcher(store=repository,
                                 index=vector_index,
                                 embedding_model=embedding_model)
        hybrid_searcher = HybridSearcher(store=repository, bm25_index=bm25_index, searcher=searcher)

        try:
            results = run_search(
                hybrid_searcher,
                query,
                mode="hybrid",
                top_k=top_k,
                candidate_k=candidate_k,
                rerank=True,
                reranker=reranker,
            )
        except (ValueError, RuntimeError) as exc:
            console.print(f"[red]{exc}[/red]")
            continue

        display_rows = with_display_scores(results)
        _print_results(display_rows)
        console.print()
