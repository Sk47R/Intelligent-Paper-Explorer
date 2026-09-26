# Intelligent Paper Finder

**Intelligent Paper Finder** is a command-line toolkit for discovering, searching, and navigating academic papers from [arXiv](https://arxiv.org). It builds a local, persistent, incrementally-updated library of papers, and searches that library using semantic embeddings, classic keyword search (BM25), a tunable hybrid of both, and optional cross-encoder reranking for higher precision.

## Motivation

Researchers and students face thousands of new papers a month. Keyword search on arXiv/Google Scholar often misses conceptually related work phrased with different terminology, while pure semantic search can under-rank exact terms, acronyms, or author names. The Intelligent Paper Finder builds a personal, local, searchable library over any topic you choose, and lets you search it by meaning, by exact keyword, or both, with results you can re-run instantly and offline, without re-querying arXiv every time.

## What it does

- **Ingests** papers from the live arXiv API into a local SQLite database, computing an embedding for each paper once and only once.
- **Persists, tracks, and resumes** ingestion: re-running the same query skips unchanged papers, re-embeds only papers whose text have actually changed, and survives being interrupted mid-run.
- **Searches** the local library three ways: semantic (embeddings + FAISS), keyword (BM25), or a normalized hybrid of both. All three are entirely offline once papers are ingested.
- **Reranks** search candidates with a local cross-encoder for higher precision, without ever scanning the full corpus.
- **Filters** search results by arXiv category and publication date.
- **Interactively explores**: type a query, and it auto-ingests from arXiv only if your local library doesn't already cover the topic well, then searches with hybrid+rerank and shows meaningfully-scaled (0-1) scores.

## Installation

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo-url>
cd paper-explorer
uv pip install -e .
```

Verify the install:

```bash
uv run -m paper_explorer --help
```

> Note: The first time you run any command that embeds text, `sentence-transformers` downloads a small pretrained model (~90MB) from Hugging Face and caches it locally; the first time you use `--rerank` or `explore`, it additionally downloads a small cross-encoder model (~80MB). Both are one-time downloads. After that, everything runs fully offline.

## Quickstart

```bash
uv run -m paper_explorer ingest --query "transformer attention NLP" --max-results 100
uv run -m paper_explorer stats
uv run -m paper_explorer search "attention is all you need" --mode hybrid --rerank --top-k 10
```

Or, for a fully interactive experience that auto-ingests when needed:

```bash
uv run -m paper_explorer explore
```

## Architecture

![Architecture diagram](/Users/sujan/Desktop/paper_finder/paper_explorer/paper_explorer.png)
