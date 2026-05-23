"""Tests that don't require network or LLM keys.

Run with:
    pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arxiv_companion.ingest import Paper, chunk_paper, normalize_arxiv_id
from arxiv_companion.store import CitationGraph
from eval.metrics import (
    average_precision,
    hit_rate_at_k,
    mrr,
    precision_at_k,
    recall_at_k,
)


# ---------- metrics ----------

class TestMetrics:
    def test_hit_at_k_true(self):
        assert hit_rate_at_k(["a", "b", "c"], ["c"], 3) == 1.0

    def test_hit_at_k_false_outside_k(self):
        assert hit_rate_at_k(["a", "b", "c", "d"], ["d"], 3) == 0.0

    def test_hit_at_k_empty_expected(self):
        assert hit_rate_at_k(["a", "b"], [], 5) == 0.0

    def test_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "c", "x"], 3) == pytest.approx(2 / 3)

    def test_recall_caps_at_one(self):
        assert recall_at_k(["a", "b"], ["a", "b"], 5) == 1.0

    def test_mrr_first_hit(self):
        assert mrr(["a", "b", "c"], ["b"]) == 0.5

    def test_mrr_no_hit(self):
        assert mrr(["a", "b"], ["x"]) == 0.0

    def test_precision(self):
        assert precision_at_k(["a", "b", "c", "d"], ["a", "c"], 4) == 0.5

    def test_ap_perfect(self):
        # Both expected items at the top in order.
        assert average_precision(["a", "b", "c"], ["a", "b"]) == 1.0

    def test_ap_mixed(self):
        # Hits at ranks 1 and 3. AP = (1/1 + 2/3) / 2 = 0.833...
        result = average_precision(["a", "x", "b"], ["a", "b"])
        assert result == pytest.approx((1.0 + 2 / 3) / 2)


# ---------- citation graph ----------

def _toy_papers() -> list[Paper]:
    # A → B → C ; A → D ; E cites A and B
    # So:  A.references = [B, D]; B.references = [C]; E.references = [A, B]
    return [
        Paper(arxiv_id="A", title="A", abstract="aa", authors=[], year=2020,
              references=["B", "D"], citations=["E"]),
        Paper(arxiv_id="B", title="B", abstract="bb", authors=[], year=2018,
              references=["C"], citations=["A", "E"]),
        Paper(arxiv_id="C", title="C", abstract="cc", authors=[], year=2015,
              references=[], citations=["B"]),
        Paper(arxiv_id="D", title="D", abstract="dd", authors=[], year=2017,
              references=[], citations=["A"]),
        Paper(arxiv_id="E", title="E", abstract="ee", authors=[], year=2022,
              references=["A", "B"], citations=[]),
    ]


class TestCitationGraph:
    def test_construction_adds_all_nodes(self):
        g = CitationGraph.from_papers(_toy_papers())
        stats = g.stats()
        assert stats["nodes"] == 5
        assert stats["in_corpus"] == 5

    def test_construction_adds_edges(self):
        g = CitationGraph.from_papers(_toy_papers())
        # A cites B and D — those are 'references'.
        assert g.g.has_edge("A", "B")
        assert g.g.has_edge("A", "D")
        # E cites A and B.
        assert g.g.has_edge("E", "A")
        assert g.g.has_edge("E", "B")

    def test_ancestors_one_hop(self):
        g = CitationGraph.from_papers(_toy_papers())
        anc = g.ancestors("A", max_hops=1)
        assert set(anc.keys()) == {"B", "D"}
        assert all(v == 1 for v in anc.values())

    def test_ancestors_two_hops(self):
        g = CitationGraph.from_papers(_toy_papers())
        anc = g.ancestors("A", max_hops=2)
        # Should include C (via B) at distance 2.
        assert anc["B"] == 1
        assert anc["D"] == 1
        assert anc["C"] == 2

    def test_descendants(self):
        g = CitationGraph.from_papers(_toy_papers())
        desc = g.descendants("B", max_hops=2)
        # B is cited by A and E, A is cited by E. So at hop 1: A, E; hop 2: E.
        assert desc["A"] == 1
        assert desc["E"] == 1  # E directly cites B

    def test_pagerank_runs(self):
        g = CitationGraph.from_papers(_toy_papers())
        pr = g.compute_pagerank()
        assert set(pr.keys()) == {"A", "B", "C", "D", "E"}
        # All values positive and sum ~1.
        assert all(v > 0 for v in pr.values())
        assert sum(pr.values()) == pytest.approx(1.0, abs=1e-6)

    def test_pagerank_ranks_highly_cited_higher(self):
        # In our toy graph, B is cited by both A and E and is also pointed to
        # by C (C is itself cited only by B). With reversed graph for PageRank,
        # B should outrank D (which is only cited by A).
        g = CitationGraph.from_papers(_toy_papers())
        pr = g.compute_pagerank()
        assert pr["B"] > pr["D"]

    def test_save_load_roundtrip(self, tmp_path):
        g = CitationGraph.from_papers(_toy_papers())
        g.compute_pagerank()
        p = tmp_path / "graph.gpickle"
        g.save(p)
        g2 = CitationGraph.load(p)
        assert g2.g.number_of_nodes() == 5
        assert g2.g.number_of_edges() == g.g.number_of_edges()
        # PageRank should round-trip too.
        assert g2.pagerank("B") == pytest.approx(g.pagerank("B"))


# ---------- ingest helpers ----------

class TestIngestHelpers:
    @pytest.mark.parametrize("raw, expected", [
        ("2006.11239", "2006.11239"),
        ("http://arxiv.org/abs/2006.11239v3", "2006.11239"),
        ("arXiv:1503.03585", "1503.03585"),
        ("1234.56789", "1234.56789"),
        ("not an arxiv id", None),
        ("", None),
    ])
    def test_normalize_arxiv_id(self, raw, expected):
        assert normalize_arxiv_id(raw) == expected

    def test_chunk_paper_produces_title_and_abstract(self):
        p = Paper(arxiv_id="2006.11239", title="DDPM", abstract="A short abstract.",
                  authors=["Ho"], year=2020)
        chunks = chunk_paper(p)
        assert len(chunks) == 2
        sections = {c.section for c in chunks}
        assert sections == {"title", "abstract"}
        assert any(c.text == "DDPM" for c in chunks)

    def test_chunk_paper_handles_empty_abstract(self):
        p = Paper(arxiv_id="X", title="T", abstract="", authors=[], year=None)
        chunks = chunk_paper(p)
        # Only the title chunk.
        assert len(chunks) == 1
        assert chunks[0].section == "title"

    def test_chunk_paper_splits_long_abstract(self):
        long_text = "x" * 5000
        p = Paper(arxiv_id="X", title="T", abstract=long_text, authors=[], year=None)
        chunks = chunk_paper(p, chunk_size=1200, overlap=200)
        # Title + multiple abstract chunks.
        abs_chunks = [c for c in chunks if c.section == "abstract"]
        assert len(abs_chunks) > 1
