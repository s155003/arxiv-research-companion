"""Data ingestion: arXiv + Semantic Scholar.

Responsibilities:
- Search arXiv for papers matching a topic query.
- For each paper, fetch references and citations from Semantic Scholar
  (keyed by arXiv ID), which is what powers the citation graph.
- Cache raw API responses on disk so re-runs are cheap.
- Chunk paper text (abstract + any available sections) for embedding.

Why Semantic Scholar? arXiv itself doesn't expose citation edges. S2 does,
indexed by arXiv ID, with a generous free tier.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator

import arxiv
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "paperId,externalIds,title,abstract,year,authors,references.externalIds,citations.externalIds"


@dataclass
class Paper:
    """Normalized record we store and reason about everywhere downstream.

    `arxiv_id` is the canonical identifier — it's stable, dereferenceable,
    and works as a key in both the vector store and the citation graph.
    """

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int | None
    categories: list[str] = field(default_factory=list)
    # Citation graph edges, as lists of arXiv IDs we successfully resolved.
    references: list[str] = field(default_factory=list)  # papers this one cites
    citations: list[str] = field(default_factory=list)  # papers that cite this one
    s2_id: str | None = None

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Paper":
        return cls(**d)


# ---------- arXiv ----------

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def normalize_arxiv_id(raw: str) -> str | None:
    """Strip URL prefixes / version suffixes to get a bare arXiv ID like '2006.11239'."""
    if not raw:
        return None
    m = _ARXIV_ID_RE.search(raw)
    return m.group(1) if m else None


def search_arxiv(query: str, max_results: int = 100) -> Iterator[arxiv.Result]:
    """Yield arXiv search results. Uses the official API via the `arxiv` package."""
    client = arxiv.Client(page_size=10, delay_seconds=15.0, num_retries=8)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    yield from client.results(search)


def _arxiv_result_to_paper(r: arxiv.Result) -> Paper:
    aid = normalize_arxiv_id(r.entry_id) or r.get_short_id()
    return Paper(
        arxiv_id=aid.split("v")[0],  # drop any version suffix
        title=r.title.strip(),
        abstract=(r.summary or "").strip(),
        authors=[a.name for a in r.authors],
        year=r.published.year if r.published else None,
        categories=list(r.categories or []),
    )


# ---------- Semantic Scholar ----------

class SemanticScholarClient:
    """Thin client with manual rate limiting and on-disk caching."""

    def __init__(
        self,
        cache_dir: Path,
        rps: float = 1.0,
        api_key: str | None = None,
    ):
        self.cache_dir = Path(cache_dir) / "s2"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / max(rps, 0.01)
        self._last_call = 0.0
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    def fetch_by_arxiv_id(self, arxiv_id: str) -> dict | None:
        """Return the S2 paper record for an arXiv ID, or None on miss."""
        cache_path = self.cache_dir / f"{arxiv_id}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        self._throttle()
        url = f"{S2_BASE}/paper/arXiv:{arxiv_id}"
        try:
            resp = self.session.get(url, params={"fields": S2_FIELDS}, timeout=30)
        except requests.RequestException as e:
            log.warning("S2 request failed for %s: %s", arxiv_id, e)
            return None

        if resp.status_code == 429:
            # Backoff and retry once.
            log.warning("S2 rate-limited; backing off 10s")
            time.sleep(10)
            resp = self.session.get(url, params={"fields": S2_FIELDS}, timeout=30)

        if resp.status_code != 200:
            log.info("S2 miss %s for %s", resp.status_code, arxiv_id)
            return None

        data = resp.json()
        cache_path.write_text(json.dumps(data))
        return data


def _extract_arxiv_refs(s2_record: dict, key: str) -> list[str]:
    """Pull arXiv IDs out of S2's `references` or `citations` blocks.

    Many S2 papers aren't on arXiv — we drop those. The graph is intentionally
    arXiv-only so every node is dereferenceable and re-fetchable.
    """
    out: list[str] = []
    for item in s2_record.get(key) or []:
        ext = (item or {}).get("externalIds") or {}
        aid = ext.get("ArXiv")
        if aid:
            norm = normalize_arxiv_id(aid)
            if norm:
                out.append(norm)
    return out


# ---------- High-level orchestration ----------

def fetch_papers(
    query: str,
    max_papers: int,
    cache_dir: Path,
    s2_rps: float = 1.0,
    s2_api_key: str | None = None,
) -> list[Paper]:
    """Pipeline: arXiv search → S2 enrichment → Paper records.

    Idempotent over the cache directory: rerunning with the same query is
    just file reads after the first pass.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arxiv_cache = cache_dir / "arxiv.jsonl"

    # Step 1: arXiv search (with a tiny cache so we don't re-paginate).
    papers: list[Paper] = []
    if arxiv_cache.exists():
        with open(arxiv_cache) as f:
            cached = [Paper.from_json(json.loads(line)) for line in f]
        if len(cached) >= max_papers:
            papers = cached[:max_papers]

    if not papers:
        log.info("Searching arXiv: %r", query)
        for r in tqdm(search_arxiv(query, max_papers), total=max_papers, desc="arxiv"):
            papers.append(_arxiv_result_to_paper(r))
        with open(arxiv_cache, "w") as f:
            for p in papers:
                f.write(json.dumps(p.to_json()) + "\n")

    # Step 2: enrich with S2 citation edges.
    s2 = SemanticScholarClient(cache_dir, rps=s2_rps, api_key=s2_api_key)
    for p in tqdm(papers, desc="s2"):
        rec = s2.fetch_by_arxiv_id(p.arxiv_id)
        if not rec:
            continue
        p.s2_id = rec.get("paperId")
        p.references = _extract_arxiv_refs(rec, "references")
        p.citations = _extract_arxiv_refs(rec, "citations")

    return papers


# ---------- Chunking ----------

@dataclass
class Chunk:
    chunk_id: str
    arxiv_id: str
    text: str
    # 'abstract' | 'title' | 'body:<section>' — used so we can boost certain sections.
    section: str = "abstract"


def chunk_paper(paper: Paper, chunk_size: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Produce embedding-sized chunks for a paper.

    Today this only chunks title+abstract — sufficient for a strong baseline
    when full text isn't available. Extend this to walk full-text sections
    when you ingest PDFs.
    """
    parts: list[Chunk] = []
    # Title gets its own chunk so it can match short, lexical queries.
    parts.append(Chunk(
        chunk_id=f"{paper.arxiv_id}::title",
        arxiv_id=paper.arxiv_id,
        text=paper.title,
        section="title",
    ))
    if paper.abstract:
        text = paper.abstract
        if len(text) <= chunk_size:
            parts.append(Chunk(
                chunk_id=f"{paper.arxiv_id}::abs",
                arxiv_id=paper.arxiv_id,
                text=text,
                section="abstract",
            ))
        else:
            # Sliding window — abstracts are short enough this rarely triggers,
            # but the same logic generalizes to full-text sections.
            start = 0
            i = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                parts.append(Chunk(
                    chunk_id=f"{paper.arxiv_id}::abs::{i}",
                    arxiv_id=paper.arxiv_id,
                    text=text[start:end],
                    section="abstract",
                ))
                if end == len(text):
                    break
                start = end - overlap
                i += 1
    return parts


def chunk_all(papers: Iterable[Paper], chunk_size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for p in papers:
        out.extend(chunk_paper(p, chunk_size=chunk_size, overlap=overlap))
    return out
