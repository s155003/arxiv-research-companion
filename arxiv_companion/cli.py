"""Command-line entry points. Most user-facing operations route through here."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from . import load_config
from .ingest import chunk_all, fetch_papers
from .generation import answer, build_generator
from .retrieval import (
    BM25Retriever,
    Embedder,
    HybridRetriever,
    HybridWeights,
    VectorRetriever,
)
from .store import CitationGraph, VectorStore

load_dotenv()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("arxiv_companion")


@click.group()
def cli():
    """arXiv Research Companion."""


# ---------- build-index ----------

@cli.command("build-index")
@click.option("--query", required=True, help="arXiv search query, e.g. 'diffusion models'.")
@click.option("--max-papers", type=int, default=200, show_default=True)
@click.option("--config", "config_path", type=click.Path(), default=None)
def build_index_cmd(query: str, max_papers: int, config_path: str | None):
    """Fetch papers, build the citation graph, and index embeddings."""
    cfg = load_config(config_path)

    # 1. Ingest
    papers = fetch_papers(
        query=query,
        max_papers=max_papers,
        cache_dir=Path(cfg["ingest"]["cache_dir"]),
        s2_rps=cfg["citation_graph"].get("semantic_scholar_rps", 1.0),
        s2_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
    )
    log.info("Fetched %d papers", len(papers))

    # 2. Build + persist citation graph
    graph = CitationGraph.from_papers(papers)
    graph.compute_pagerank(alpha=cfg["citation_graph"].get("pagerank_alpha", 0.85))
    graph.save(cfg["citation_graph"]["path"])
    log.info("Citation graph: %s", graph.stats())

    # 3. Chunk + embed + persist vectors
    chunks = chunk_all(
        papers,
        chunk_size=cfg["ingest"]["chunk_size"],
        overlap=cfg["ingest"]["chunk_overlap"],
    )
    log.info("Embedding %d chunks", len(chunks))
    embedder = Embedder(
        cfg["embeddings"]["model"],
        normalize=cfg["embeddings"].get("normalize", True),
        batch_size=cfg["embeddings"].get("batch_size", 64),
    )
    embs = embedder.embed([c.text for c in chunks])
    vstore = VectorStore(
        cfg["vector_store"]["path"],
        cfg["vector_store"]["collection"],
        distance=cfg["vector_store"].get("distance", "cosine"),
    )
    vstore.upsert(chunks, embs)
    log.info("Vector store contains %d chunks", vstore.count())


# ---------- query ----------

@cli.command("query")
@click.argument("question", nargs=-1, required=True)
@click.option("--retriever", type=click.Choice(["vector", "bm25", "hybrid"]), default="hybrid")
@click.option("--k", type=int, default=10, show_default=True)
@click.option("--config", "config_path", type=click.Path(), default=None)
@click.option("--json-out", is_flag=True, help="Emit JSON instead of formatted text.")
@click.option("--no-llm", is_flag=True, help="Skip generation; just print retrieved papers.")
def query_cmd(question, retriever, k, config_path, json_out, no_llm):
    """Answer a question using the indexed corpus."""
    q = " ".join(question)
    cfg = load_config(config_path)

    graph = CitationGraph.load(cfg["citation_graph"]["path"])
    embedder = Embedder(cfg["embeddings"]["model"],
                        normalize=cfg["embeddings"].get("normalize", True))
    vstore = VectorStore(
        cfg["vector_store"]["path"],
        cfg["vector_store"]["collection"],
        distance=cfg["vector_store"].get("distance", "cosine"),
    )

    retr = _make_retriever(retriever, cfg, vstore, graph, embedder)
    papers = _retrieve(retr, q, k, cfg)

    if no_llm:
        result_papers = papers
        answer_text = None
    else:
        gen = build_generator(cfg) if cfg["generation"]["provider"] != "none" else build_generator({"generation": {"provider": "none"}})
        out = answer(q, papers, gen)
        answer_text = out.answer
        result_papers = out.used_papers

    if json_out:
        click.echo(json.dumps({
            "question": q,
            "retriever": retriever,
            "answer": answer_text,
            "papers": [
                {
                    "rank": i + 1,
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "year": p.year,
                    "score": p.score,
                    "semantic_sim": p.semantic_sim,
                    "pagerank": p.pagerank,
                    "hop_distance": p.hop_distance,
                    "source": p.source,
                    "url": f"https://arxiv.org/abs/{p.arxiv_id}",
                }
                for i, p in enumerate(result_papers)
            ],
        }, indent=2))
        return

    _print_human(q, retriever, answer_text, result_papers)


def _make_retriever(name, cfg, vstore, graph, embedder):
    if name == "vector":
        return VectorRetriever(vstore, embedder, graph=graph)
    if name == "bm25":
        # BM25 needs the original Paper objects. Reconstruct from the graph
        # nodes that are in-corpus (cheap approximation; for richer text,
        # cache Paper objects alongside the graph in a future revision).
        from .ingest import Paper as _Paper
        papers = []
        for nid, attrs in graph.g.nodes(data=True):
            if attrs.get("in_corpus"):
                papers.append(_Paper(
                    arxiv_id=nid,
                    title=attrs.get("title", ""),
                    abstract="",  # not stored in graph; BM25 falls back to titles
                    authors=attrs.get("authors", []) or [],
                    year=attrs.get("year"),
                ))
        return BM25Retriever(papers, graph=graph)
    # hybrid
    w = cfg["retrieval"]["weights"]
    return HybridRetriever(
        vstore=vstore,
        graph=graph,
        embedder=embedder,
        weights=HybridWeights(
            alpha=w["alpha"], beta=w["beta"], gamma=w["gamma"], delta=w["delta"],
        ),
        max_hops=cfg["citation_graph"].get("max_hops", 2),
    )


def _retrieve(retr, query, k, cfg):
    if isinstance(retr, HybridRetriever):
        return retr.retrieve(
            query, k=k,
            k_seed=cfg["retrieval"]["k_seed"],
            k_expand=cfg["retrieval"]["k_expand"],
        )
    return retr.retrieve(query, k=k)


def _print_human(question, retriever_name, answer_text, papers):
    click.secho(f"\nQuestion: {question}", fg="cyan", bold=True)
    click.secho(f"Retriever: {retriever_name}\n", fg="cyan")
    if answer_text:
        click.echo(answer_text)
        click.echo("\n" + "─" * 70)
    click.secho("Retrieved papers:", bold=True)
    for i, p in enumerate(papers, start=1):
        yr = f" ({p.year})" if p.year else ""
        title = p.title or "(no title)"
        click.echo(f"  [{i:>2}] arXiv:{p.arxiv_id} — {title}{yr}")
        click.secho(f"        {p.explain()}", fg="bright_black")


if __name__ == "__main__":
    cli()
