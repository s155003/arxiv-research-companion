"""Retrieval strategies.

Three retrievers, all returning a ranked list of papers (by arXiv ID):

- VectorRetriever:  dense embeddings only (the baseline most demos stop at)
- BM25Retriever:    lexical baseline
- HybridRetriever:  vector seeds → citation-graph expansion → fused rerank

The hybrid retriever is the differentiator. It uses the vector store to find
*semantically* relevant papers, then walks the citation graph from each hit
to find ancestors and descendants — papers that may not look similar in
embedding space but are causally connected to the topic.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Paper
from .store import CitationGraph, VectorStore

log = logging.getLogger(__name__)


# ---------- Result types ----------

@dataclass
class RetrievedPaper:
    arxiv_id: str
    score: float
    title: str = ""
    year: int | None = None
    # Diagnostics — useful for understanding *why* a paper ranked where it did.
    semantic_sim: float = 0.0      # max over its chunks
    pagerank: float = 0.0
    hop_distance: int | None = None   # None == direct semantic hit, 0 == seed itself
    matched_chunks: list[str] = field(default_factory=list)
    # 'semantic' | 'ancestor' | 'descendant' — primary reason it surfaced
    source: str = "semantic"

    def explain(self) -> str:
        parts = [f"semantic={self.semantic_sim:.3f}", f"pagerank={self.pagerank:.4f}"]
        if self.hop_distance is not None:
            parts.append(f"hops={self.hop_distance}")
        parts.append(f"via={self.source}")
        return " ".join(parts)


class Retriever(Protocol):
    name: str
    def retrieve(self, query: str, k: int) -> list[RetrievedPaper]: ...


# ---------- Embedder ----------

class Embedder:
    """Wraps a sentence-transformers model. One instance per process."""

    def __init__(self, model_name: str, normalize: bool = True, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedder: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ---------- Vector-only baseline ----------

class VectorRetriever:
    """Dense retrieval, deduped to paper level by taking max chunk similarity."""
    name = "vector_only"

    def __init__(self, vstore: VectorStore, embedder: Embedder, graph: CitationGraph | None = None):
        self.vstore = vstore
        self.embedder = embedder
        # Graph is optional here, used only to enrich metadata on results.
        self.graph = graph

    def retrieve(self, query: str, k: int, k_chunks: int | None = None) -> list[RetrievedPaper]:
        # Overshoot at the chunk level so we can dedupe to k distinct papers.
        k_chunks = k_chunks or max(k * 4, 40)
        q = self.embedder.embed_one(query)
        chunk_hits = self.vstore.query(q, k=k_chunks)

        best_per_paper: dict[str, RetrievedPaper] = {}
        for hit in chunk_hits:
            aid = hit["arxiv_id"]
            sim = hit["similarity"]
            existing = best_per_paper.get(aid)
            if existing is None or sim > existing.semantic_sim:
                title = self.graph.node_attr(aid, "title", "") if self.graph else ""
                year = self.graph.node_attr(aid, "year", None) if self.graph else None
                pr = self.graph.pagerank(aid) if self.graph else 0.0
                best_per_paper[aid] = RetrievedPaper(
                    arxiv_id=aid,
                    score=sim,
                    semantic_sim=sim,
                    pagerank=pr,
                    title=title,
                    year=year,
                    matched_chunks=[hit["chunk_id"]],
                    source="semantic",
                )
            else:
                existing.matched_chunks.append(hit["chunk_id"])

        ranked = sorted(best_per_paper.values(), key=lambda r: r.score, reverse=True)
        return ranked[:k]


# ---------- BM25 baseline ----------

class BM25Retriever:
    """Sparse lexical baseline over title+abstract."""
    name = "bm25"

    def __init__(self, papers: Sequence[Paper], graph: CitationGraph | None = None):
        self.papers = list(papers)
        self.graph = graph
        # Trivial whitespace tokenizer — fine for a baseline.
        self._tokenized = [self._tokenize(f"{p.title}. {p.abstract}") for p in self.papers]
        self.bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower() for t in text.split() if t]

    def retrieve(self, query: str, k: int) -> list[RetrievedPaper]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        results: list[RetrievedPaper] = []
        for idx in order:
            p = self.papers[int(idx)]
            score = float(scores[int(idx)])
            if score <= 0:
                continue
            results.append(RetrievedPaper(
                arxiv_id=p.arxiv_id,
                score=score,
                title=p.title,
                year=p.year,
                semantic_sim=0.0,
                pagerank=self.graph.pagerank(p.arxiv_id) if self.graph else 0.0,
                source="semantic",
            ))
        return results


# ---------- Hybrid: vector + citation graph ----------

@dataclass
class HybridWeights:
    alpha: float = 0.6    # semantic similarity
    beta: float = 0.3     # pagerank / influence
    gamma: float = 0.2    # seed proximity bonus
    delta: float = 0.4    # hop penalty (subtracted)


class HybridRetriever:
    """Vector seeds → citation-graph expansion → fused rerank.

    Procedure:
      1. Embed the query, get top `k_seed` papers from the vector store.
      2. From each seed, walk the citation graph up to `max_hops`:
           - successors (papers cited by seeds) ≈ foundational ancestors
           - predecessors (papers citing seeds) ≈ recent extensions
      3. Merge candidates. Each paper accumulates:
           - semantic_sim   (max chunk sim if it appeared in vector hits, else 0)
           - pagerank
           - min hop distance to any seed
           - seed proximity score = 1 / (1 + min_hop)
      4. Final score = α·sem + β·pr + γ·proximity - δ·hop_penalty

    The hop penalty makes graph-only papers fall behind semantic hits unless
    their PageRank is high — exactly what "foundational" should look like.
    """
    name = "hybrid"

    def __init__(
        self,
        vstore: VectorStore,
        graph: CitationGraph,
        embedder: Embedder,
        weights: HybridWeights | None = None,
        max_hops: int = 2,
    ):
        self.vstore = vstore
        self.graph = graph
        self.embedder = embedder
        self.weights = weights or HybridWeights()
        self.max_hops = max_hops

    def retrieve(
        self,
        query: str,
        k: int,
        k_seed: int = 20,
        k_expand: int = 30,
    ) -> list[RetrievedPaper]:
        # ---- Step 1: seeds from the vector store ----
        q = self.embedder.embed_one(query)
        chunk_hits = self.vstore.query(q, k=max(k_seed * 3, 30))

        seed_sim: dict[str, float] = {}  # arxiv_id -> best chunk sim
        seed_chunks: dict[str, list[str]] = {}
        for hit in chunk_hits:
            aid = hit["arxiv_id"]
            if aid not in seed_sim or hit["similarity"] > seed_sim[aid]:
                seed_sim[aid] = hit["similarity"]
            seed_chunks.setdefault(aid, []).append(hit["chunk_id"])

        # Top-k_seed by similarity become the actual graph seeds.
        seeds = sorted(seed_sim.items(), key=lambda x: x[1], reverse=True)[:k_seed]
        seed_ids = [aid for aid, _ in seeds]

        # ---- Step 2: graph expansion ----
        candidates: dict[str, RetrievedPaper] = {}

        # Seed papers themselves go in first.
        for aid in seed_ids:
            sim = seed_sim[aid]
            candidates[aid] = RetrievedPaper(
                arxiv_id=aid,
                score=0.0,
                semantic_sim=sim,
                pagerank=self.graph.pagerank(aid),
                title=self.graph.node_attr(aid, "title", "") or "",
                year=self.graph.node_attr(aid, "year", None),
                hop_distance=0,
                matched_chunks=seed_chunks.get(aid, []),
                source="semantic",
            )

        for seed_id in seed_ids:
            # Ancestors: papers this seed cites (the "foundational" direction).
            ancestors = self.graph.ancestors(seed_id, max_hops=self.max_hops)
            self._merge_graph_hits(
                candidates, ancestors, seed_id, seed_sim,
                k_expand=k_expand, source="ancestor",
            )
            # Descendants: papers that cite this seed.
            descendants = self.graph.descendants(seed_id, max_hops=self.max_hops)
            self._merge_graph_hits(
                candidates, descendants, seed_id, seed_sim,
                k_expand=k_expand, source="descendant",
            )

        # ---- Step 3: fused scoring ----
        w = self.weights
        for r in candidates.values():
            proximity = 1.0 / (1.0 + (r.hop_distance or 0))
            hop_pen = (r.hop_distance or 0) * 1.0
            r.score = (
                w.alpha * r.semantic_sim
                + w.beta * _scale_pagerank(r.pagerank)
                + w.gamma * proximity
                - w.delta * (hop_pen if r.semantic_sim == 0 else 0.0)
                # ^ only penalize hops for graph-only papers; seeds keep full credit
            )

        ranked = sorted(candidates.values(), key=lambda r: r.score, reverse=True)
        return ranked[:k]

    def _merge_graph_hits(
        self,
        candidates: dict[str, RetrievedPaper],
        hits: dict[str, int],
        seed_id: str,
        seed_sim: dict[str, float],
        k_expand: int,
        source: str,
    ) -> None:
        """Insert/update graph candidates while tracking min hop distance.

        We cap per-seed expansions at k_expand by PageRank to avoid one seed
        dragging in a huge fan-out of weakly-relevant papers.
        """
        if not hits:
            return
        # Rank graph hits by pagerank so the most influential papers per
        # neighborhood get priority when we hit the k_expand cap.
        ranked_hits = sorted(
            hits.items(),
            key=lambda x: self.graph.pagerank(x[0]),
            reverse=True,
        )[:k_expand]

        for aid, hop in ranked_hits:
            existing = candidates.get(aid)
            sim = seed_sim.get(aid, 0.0)  # could also be a direct vector hit
            if existing is None:
                candidates[aid] = RetrievedPaper(
                    arxiv_id=aid,
                    score=0.0,
                    semantic_sim=sim,
                    pagerank=self.graph.pagerank(aid),
                    title=self.graph.node_attr(aid, "title", "") or "",
                    year=self.graph.node_attr(aid, "year", None),
                    hop_distance=hop,
                    source=source,
                )
            else:
                # Keep the minimum hop distance across all paths to this paper.
                if existing.hop_distance is None or hop < existing.hop_distance:
                    existing.hop_distance = hop
                # If the new path is via a different source type, prefer the
                # one that gives the paper its primary identity in results.
                # Semantic hits always win; otherwise keep first source.


def _scale_pagerank(pr: float) -> float:
    """PageRank values are tiny (often 1e-4). Log-scale so the β weight
    operates on a number comparable to similarity (0..1)."""
    if pr <= 0:
        return 0.0
    return min(1.0, math.log1p(pr * 10_000) / math.log(10_000))
