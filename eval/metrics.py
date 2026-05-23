"""Evaluation metrics for ranked retrieval.

All functions take a list of *retrieved* arXiv IDs and a list of *expected*
arXiv IDs (the golden set for a question) and return a scalar. The
foundational-restricted variants take a set of arxiv IDs that share a given
tag (e.g. 'foundational') and ignore the rest.
"""
from __future__ import annotations

from collections.abc import Iterable


def hit_rate_at_k(retrieved: list[str], expected: Iterable[str], k: int) -> float:
    """1.0 if any expected paper is in top-k, else 0.0."""
    if not expected:
        return 0.0
    top = set(retrieved[:k])
    return 1.0 if top.intersection(expected) else 0.0


def recall_at_k(retrieved: list[str], expected: Iterable[str], k: int) -> float:
    """Fraction of expected papers present in the top-k."""
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    top = set(retrieved[:k])
    return len(top.intersection(expected_set)) / len(expected_set)


def mrr(retrieved: list[str], expected: Iterable[str]) -> float:
    """Mean reciprocal rank of the first expected paper. Rank is 1-indexed."""
    expected_set = set(expected)
    for i, aid in enumerate(retrieved, start=1):
        if aid in expected_set:
            return 1.0 / i
    return 0.0


def precision_at_k(retrieved: list[str], expected: Iterable[str], k: int) -> float:
    """Fraction of top-k that are expected."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    expected_set = set(expected)
    return sum(1 for r in top if r in expected_set) / len(top)


def average_precision(retrieved: list[str], expected: Iterable[str]) -> float:
    """Average precision over the retrieved list. Ignores expected papers
    that never appear in the ranking (their contribution is 0)."""
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    hits = 0
    summed = 0.0
    for i, aid in enumerate(retrieved, start=1):
        if aid in expected_set:
            hits += 1
            summed += hits / i
    if hits == 0:
        return 0.0
    return summed / len(expected_set)
