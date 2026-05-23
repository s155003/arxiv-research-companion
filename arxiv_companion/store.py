"""Persistence: ChromaDB for vectors, NetworkX for the citation graph.

These are split intentionally — they have different access patterns and
different lifecycle (the graph is rebuilt from cached S2 data; the vector
store is rebuilt from embedding chunks).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import networkx as nx

from .ingest import Chunk, Paper

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)


# ---------- Vector store ----------

class VectorStore:
    """Wrapper around a persistent Chroma collection.

    Embedding is done by the caller (so we can swap models without
    re-instantiating Chroma's pipeline). We pass precomputed vectors.
    """

    def __init__(self, path: str | Path, collection_name: str, distance: str = "cosine"):
        import chromadb
        from chromadb.config import Settings
        self.path = str(path)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.path,
            settings=Settings(anonymized_telemetry=False),
        )
        # `get_or_create` so build_index can be re-run idempotently.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance},
        )

    def upsert(self, chunks: Sequence[Chunk], embeddings: "np.ndarray") -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have equal length")
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"arxiv_id": c.arxiv_id, "section": c.section} for c in chunks],
        )

    def query(self, embedding: "np.ndarray", k: int) -> list[dict]:
        """Return list of {chunk_id, arxiv_id, section, text, distance}."""
        res = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append({
                "chunk_id": cid,
                "arxiv_id": meta.get("arxiv_id"),
                "section": meta.get("section"),
                "text": doc,
                "distance": float(dist),
                # For cosine distance in Chroma, similarity = 1 - distance.
                "similarity": 1.0 - float(dist),
            })
        return out

    def count(self) -> int:
        return self.collection.count()


# ---------- Citation graph ----------

class CitationGraph:
    """Directed citation graph: edge u -> v means paper u cites paper v.

    Nodes are arXiv IDs. Node attributes include title/year so we don't
    need to round-trip through the Paper store on retrieval.
    """

    def __init__(self, graph: nx.DiGraph | None = None):
        self.g: nx.DiGraph = graph if graph is not None else nx.DiGraph()
        self._pagerank: dict[str, float] = {}

    # ---- build ----

    @classmethod
    def from_papers(cls, papers: Iterable[Paper]) -> "CitationGraph":
        g = nx.DiGraph()
        papers = list(papers)
        # First pass: add all nodes we know about.
        for p in papers:
            g.add_node(
                p.arxiv_id,
                title=p.title,
                year=p.year,
                authors=p.authors,
                in_corpus=True,
            )
        # Second pass: edges. References from neighbors that aren't in our
        # corpus still get added as nodes so we can traverse to them.
        known = {p.arxiv_id for p in papers}
        for p in papers:
            for ref in p.references:
                if ref not in g:
                    g.add_node(ref, in_corpus=ref in known)
                g.add_edge(p.arxiv_id, ref)
            for cit in p.citations:
                if cit not in g:
                    g.add_node(cit, in_corpus=cit in known)
                g.add_edge(cit, p.arxiv_id)
        return cls(g)

    # ---- persistence ----

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"graph": self.g, "pagerank": self._pagerank}, f)

    @classmethod
    def load(cls, path: str | Path) -> "CitationGraph":
        with open(path, "rb") as f:
            d = pickle.load(f)
        cg = cls(d["graph"])
        cg._pagerank = d.get("pagerank") or {}
        return cg

    # ---- analytics ----

    def compute_pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """PageRank over the inverse graph: influence flows from descendants
        back to ancestors. We reverse so that highly-cited papers get high
        scores (the standard interpretation)."""
        if self.g.number_of_nodes() == 0:
            self._pagerank = {}
            return self._pagerank
        reverse = self.g.reverse(copy=False)
        self._pagerank = nx.pagerank(reverse, alpha=alpha)
        return self._pagerank

    def pagerank(self, node: str) -> float:
        return self._pagerank.get(node, 0.0)

    # ---- queries ----

    def ancestors(self, node: str, max_hops: int = 2) -> dict[str, int]:
        """Papers reachable by following 'cites' edges from node.

        Returns {arxiv_id: hop_distance}. Hop 1 = directly cited.
        """
        return self._bfs(node, max_hops, direction="out")

    def descendants(self, node: str, max_hops: int = 2) -> dict[str, int]:
        """Papers that cite this one (and their citers), up to max_hops."""
        return self._bfs(node, max_hops, direction="in")

    def _bfs(self, start: str, max_hops: int, direction: str) -> dict[str, int]:
        if start not in self.g:
            return {}
        # BFS so we get the minimum hop distance for each reachable node.
        visited: dict[str, int] = {start: 0}
        frontier = [start]
        for hop in range(1, max_hops + 1):
            next_frontier = []
            for node in frontier:
                neighbors = (
                    self.g.successors(node) if direction == "out"
                    else self.g.predecessors(node)
                )
                for nb in neighbors:
                    if nb not in visited:
                        visited[nb] = hop
                        next_frontier.append(nb)
            frontier = next_frontier
            if not frontier:
                break
        visited.pop(start, None)
        return visited

    def node_attr(self, node: str, key: str, default=None):
        if node not in self.g:
            return default
        return self.g.nodes[node].get(key, default)

    def has(self, node: str) -> bool:
        return node in self.g

    def stats(self) -> dict:
        in_corpus = sum(1 for _, d in self.g.nodes(data=True) if d.get("in_corpus"))
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "in_corpus": in_corpus,
            "external": self.g.number_of_nodes() - in_corpus,
            "has_pagerank": bool(self._pagerank),
        }
