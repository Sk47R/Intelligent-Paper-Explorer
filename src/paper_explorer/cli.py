from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from paper_explorer.config import (
    DEFAULT_DB_PATH,
    DEFAULT_ID_MAP_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_PLOTS_DIR,
    DEFAULT_STORE_PATH,
    RAW_DIR,
)
from paper_explorer.crawler.arxiv_client import ArxivAPIError, ArxivClient
from paper_explorer.data.repository import PaperRepository
from paper_explorer.data.storage import PaperStore
from paper_explorer.explore import DEFAULT_COVERAGE_THRESHOLD, run_explore
from paper_explorer.ingestion.service import IngestionService
from paper_explorer.search.bm25_index import BM25Index
from paper_explorer.search.hybrid import HybridSearcher
from paper_explorer.search.index import VectorIndex
from paper_explorer.search.pipeline import run_search
from paper_explorer.search.reranker import CrossEncoderReranker, RerankerLoadError
from paper_explorer.search.results import SearchResult
from paper_explorer.search.searcher import PaperSearcher
from paper_explorer.visualize import run_visualize

console = Console()
logger = logging.getLogger("paper_explorer")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
        force=True,
    )

    for noisy_logger in (
        "httpx",
        "httpcore",
        "urllib3",
        "huggingface_hub",
        "sentence_transformers",
        "filelock",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _load_vector_index(args: argparse.Namespace) -> VectorIndex:
    try:
        return VectorIndex.load(args.index, args.id_map)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        console.print(
            f"[red]No FAISS index found at {args.index}. "
            f"Run `ingest` or `index rebuild` first.[/red]"
        )
        raise SystemExit(1) from exc


def cmd_ingest(args: argparse.Namespace) -> None:
    repository = PaperRepository(args.db)
    service = IngestionService(repository, client=ArxivClient())

    console.print(f"[bold]Querying arXiv for[/bold] {args.query!r} ...")
    try:
        summary = service.ingest(
            args.query,
            max_results=args.max_results,
            start=args.start,
            category=args.category,
            raw_dir=None if args.no_raw else args.raw_dir,
        )
    except ArxivAPIError as exc:
        console.print(f"[red]arXiv API error:[/red] {exc}")
        raise SystemExit(1) from exc
    except ValueError as exc:
        console.print(f"[red]Invalid arguments:[/red] {exc}")
        raise SystemExit(1) from exc

    console.print(
        f"[green]Fetched {summary['fetched']}[/green] "
        f"(new={summary['new']}, changed={summary['changed']}, unchanged={summary['unchanged']})"
    )
    console.print(
        f"[green]Embedded {summary['embedded']}[/green] (failed={summary['embedding_failed']})"
    )

    index = service.rebuild_index()
    if len(index) > 0:
        index.save(args.index, args.id_map)
        console.print(f"[green]Saved FAISS index[/green] -> {args.index} ({len(index)} vectors)")
    else:
        console.print("[yellow]No embedded papers yet; index not saved.[/yellow]")


def cmd_explore(args: argparse.Namespace) -> None:
    run_explore(
        db_path=args.db,
        index_path=args.index,
        id_map_path=args.id_map,
        threshold=args.threshold,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )


def cmd_migrate(args: argparse.Namespace) -> None:
    store = PaperStore(args.from_json)
    if len(store) == 0:
        console.print(f"[yellow]No papers found in {args.from_json}[/yellow]")
        return

    repository = PaperRepository(args.db)
    migrated = 0
    embedded = 0
    for paper in store.all():
        repository.upsert_paper(paper)
        if paper.embedding is not None:
            repository.set_embedding(
                paper.paper_id, paper.embedding, model_name="unknown (migrated)"
            )
            embedded += 1
        migrated += 1

    console.print(
        f"[green]Migrated {migrated} papers[/green] from {args.from_json} -> {args.db} "
        f"({embedded} with pre-existing embeddings)"
    )

    service = IngestionService(repository, client=ArxivClient())
    index = service.rebuild_index()
    if len(index) > 0:
        index.save(args.index, args.id_map)
        console.print(f"[green]Rebuilt FAISS index[/green] -> {args.index} ({len(index)} vectors)")


def cmd_visualize(args: argparse.Namespace) -> None:
    run_visualize(
        db_path=args.db,
        index_path=args.index,
        id_map_path=args.id_map,
        query=args.query,
        output_dir=args.output_dir,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        alpha=args.alpha,
    )

def cmd_index_rebuild(args: argparse.Namespace) -> None:
    repository = PaperRepository(args.db)
    service = IngestionService(repository, client=ArxivClient())
    index = service.rebuild_index()
    if len(index) == 0:
        console.print("[yellow]No embedded papers found; nothing to index.[/yellow]")
        return
    index.save(args.index, args.id_map)
    console.print(f"[green]Rebuilt FAISS index[/green] -> {args.index} ({len(index)} vectors)")


def cmd_stats(args: argparse.Namespace) -> None:
    repository = PaperRepository(args.db)
    s = repository.stats()

    table = Table(title="Paper Explorer -- Database Statistics")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total papers", str(s["total_papers"]))
    table.add_row("Embedded", str(s["embedded"]))
    table.add_row("Pending embedding", str(s["pending"]))
    table.add_row("Stale (needs re-embedding)", str(s["stale"]))
    table.add_row("Failed", str(s["failed"]))
    table.add_row("Embedding model", str(s["embedding_model"]))
    table.add_row("Embedding dimension", str(s["embedding_dimension"]))
    table.add_row("Categories", ", ".join(s["categories"]) or "n/a")
    table.add_row("Date range", f"{s['date_range'][0]} to {s['date_range'][1]}")

    try:
        index = VectorIndex.load(args.index, args.id_map)
        table.add_row("Indexed vectors (FAISS)", str(len(index)))
    except (FileNotFoundError, OSError, RuntimeError):
        table.add_row("Indexed vectors (FAISS)", "no index file found")

    console.print(table)


def cmd_reset(args: argparse.Namespace) -> None:
    if not args.yes:
        console.print(
            "[yellow]This deletes all papers, embeddings, and the FAISS index. "
            "Re-run with --yes to confirm.[/yellow]"
        )
        raise SystemExit(1)
    repository = PaperRepository(args.db)
    repository.reset()
    for path in (Path(args.index), Path(args.id_map)):
        if path.exists():
            path.unlink()
    console.print("[green]Database and index reset.[/green]")


def cmd_search(args: argparse.Namespace) -> None:
    repository = PaperRepository(args.db)
    if len(repository) == 0:
        console.print("[red]No papers in database. Run `ingest` first.[/red]")
        raise SystemExit(1)

    papers = repository.all(category=args.category, from_date=args.from_date, to_date=args.to_date)
    if not papers:
        console.print("[yellow]No papers match the given filters.[/yellow]")
        return

    bm25_index = BM25Index()
    bm25_index.build(papers)

    semantic_searcher = None
    if args.mode in ("semantic", "hybrid"):
        vector_index = _load_vector_index(args)
        semantic_searcher = PaperSearcher(store=repository, index=vector_index)

    hybrid_searcher = HybridSearcher(
        store=repository, bm25_index=bm25_index, searcher=semantic_searcher
    )

    reranker = None
    if args.rerank:
        try:
            reranker = CrossEncoderReranker()
        except RerankerLoadError as exc:
            console.print(f"[red]Failed to load reranker:[/red] {exc}")
            raise SystemExit(1) from exc

    try:
        results = run_search(
            hybrid_searcher,
            args.query,
            mode=args.mode,
            top_k=args.top_k,
            alpha=args.alpha,
            candidate_k=args.candidate_k,
            rerank=args.rerank,
            reranker=reranker,
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc

    _print_results(results, mode=args.mode, reranked=args.rerank)


def _print_results(
    results: list[SearchResult], mode: str = "semantic", reranked: bool = False
) -> None:
    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    show_breakdown = mode == "hybrid"
    title = f"Search Results ({mode}{' + reranked' if reranked else ''})"

    table = Table(title=title)
    table.add_column("Rank", justify="right")
    if show_breakdown:
        table.add_column("Semantic", justify="right")
        table.add_column("Keyword", justify="right")
        table.add_column("Retrieval", justify="right")
    else:
        table.add_column("Retrieval", justify="right")
    if reranked:
        table.add_column("Rerank", justify="right")
    table.add_column("arXiv ID")
    table.add_column("Title")
    table.add_column("Published")

    for r in results:
        d = r.to_display_dict()
        row = [str(d["rank"])]
        if show_breakdown:
            row += [d.get("semantic_score", "-"), d.get("keyword_score", "-"), d["score"]]
        else:
            row += [d["score"]]
        if reranked:
            row += [d.get("rerank_score", "-")]
        row += [d["arxiv_id"], d["title"], d["published"]]
        table.add_row(*row)
    console.print(table)

    for r in results:
        line = f"\n[bold]{r.rank}. {r.paper.title}[/bold]  (retrieval: {r.score:.3f}"
        if r.rerank_score is not None:
            line += f", rerank: {r.rerank_score:.3f}"
        line += ")"
        console.print(line)
        console.print(f"   Authors: {', '.join(r.paper.authors) or 'n/a'}")
        console.print(f"   {r.paper.abstract_url}")
        console.print(f"   {r.short_abstract()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper_explorer", description="Intelligent Research Paper Explorer (Version 4)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the SQLite database")
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH), help="Path to the FAISS index")
    parser.add_argument(
        "--id-map", default=str(DEFAULT_ID_MAP_PATH), help="Path to the FAISS id-map"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser(
        "ingest", help="Crawl arXiv, incrementally embed, and rebuild the FAISS index"
    )
    p_ingest.add_argument("--query", required=True, help="Free-text arXiv search query")
    p_ingest.add_argument("--max-results", type=int, default=100)
    p_ingest.add_argument("--start", type=int, default=0)
    p_ingest.add_argument("--category", default=None, help="Optional arXiv category, e.g. cs.LG")
    p_ingest.add_argument("--raw-dir", default=str(RAW_DIR))
    p_ingest.add_argument("--no-raw", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_migrate = subparsers.add_parser("migrate", help="Import an existing papers.json into SQLite")
    p_migrate.add_argument("--from-json", default=str(DEFAULT_STORE_PATH))
    p_migrate.set_defaults(func=cmd_migrate)

    p_index = subparsers.add_parser("index", help="Vector index maintenance")
    index_subparsers = p_index.add_subparsers(dest="index_command", required=True)
    p_index_rebuild = index_subparsers.add_parser(
        "rebuild", help="Rebuild the FAISS index from persisted embeddings (no re-embedding)"
    )
    p_index_rebuild.set_defaults(func=cmd_index_rebuild)

    p_stats = subparsers.add_parser("stats", help="Show database/index statistics")
    p_stats.set_defaults(func=cmd_stats)

    p_reset = subparsers.add_parser("reset", help="Delete all papers, embeddings, and the index")
    p_reset.add_argument("--yes", action="store_true", help="Confirm the reset")
    p_reset.set_defaults(func=cmd_reset)

    p_search = subparsers.add_parser(
        "search", help="Search: semantic, keyword (BM25), or hybrid; optional rerank/filters"
    )
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--mode", choices=["semantic", "keyword", "hybrid"], default="semantic")
    p_search.add_argument("--alpha", type=float, default=0.5)
    p_search.add_argument("--candidate-k", type=int, default=50)
    p_search.add_argument("--rerank", action="store_true")
    p_search.add_argument("--category", default=None, help="Filter to papers in this category")
    p_search.add_argument("--from-date", default=None, help="Filter: published >= this ISO date")
    p_search.add_argument("--to-date", default=None, help="Filter: published <= this ISO date")
    p_search.set_defaults(func=cmd_search)

    p_visualize = subparsers.add_parser(
        "visualize",
        help="Generate comparison plots (PNG) for a query: mode overlap and reranking impact",
    )
    p_visualize.add_argument("query", help="Query to visualize search behavior for")
    p_visualize.add_argument("--top-k", type=int, default=10)
    p_visualize.add_argument("--candidate-k", type=int, default=50)
    p_visualize.add_argument("--alpha", type=float, default=0.5)
    p_visualize.add_argument("--output-dir", default=str(DEFAULT_PLOTS_DIR))
    p_visualize.set_defaults(func=cmd_visualize)

    p_explore = subparsers.add_parser(
        "explore",
        help="Interactive search: auto-ingests from arXiv when the local library "
        "doesn't cover a topic yet, then searches with hybrid+rerank",
    )
    p_explore.add_argument("--top-k", type=int, default=10)
    p_explore.add_argument("--candidate-k", type=int, default=50)
    p_explore.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help="Minimum local semantic score (0-1) considered 'already covered'; "
        "below this, auto-ingests 100 papers from arXiv before searching",
    )
    p_explore.set_defaults(func=cmd_explore)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
