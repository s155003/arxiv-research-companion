"""Run the golden set against all retrievers and report metrics.

Usage:
    python -m eval.evaluate
    python -m eval.evaluate --topic diffusion          # only one topic
    python -m eval.evaluate --weights alpha=0.7,beta=0.2  # sweep hybrid weights
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml

# Allow `python eval/evaluate.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arxiv_companion import load_config
from arxiv_companion.retrieval import (
    BM25Retriever,
    Embedder,
    HybridRetriever,
    HybridWeights,
    VectorRetriever,
)
from arxiv_companion.store import CitationGraph, VectorStore
from eval.metrics import (
    average_precision,
    hit_rate_at_k,
    mrr,
    recall_at_k,
)

K_VALUES = (5, 10)


def load_golden(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)


def filter_by_tag(expected: list[dict], tag: str) -> list[str]:
    return [item["id"] for item in expected if tag in (item.get("tags") or [])]


def all_expected(expected: list[dict]) -> list[str]:
    return [item["id"] for item in expected]


def evaluate_one(retriever, question: str, expected: list[dict], k_max: int) -> dict:
    """Run one retriever on one question; return per-question metrics."""
    if isinstance(retriever, HybridRetriever):
        results = retriever.retrieve(question, k=k_max, k_seed=20, k_expand=30)
    else:
        results = retriever.retrieve(question, k=k_max)
    retrieved_ids = [r.arxiv_id for r in results]
    exp_all = all_expected(expected)
    exp_found = filter_by_tag(expected, "foundational")

    out = {
        "retrieved": retrieved_ids,
        "mrr": mrr(retrieved_ids, exp_all),
        "ap": average_precision(retrieved_ids, exp_all),
    }
    for k in K_VALUES:
        out[f"hit@{k}"] = hit_rate_at_k(retrieved_ids, exp_all, k)
        out[f"recall@{k}"] = recall_at_k(retrieved_ids, exp_all, k)
        if exp_found:
            out[f"found_recall@{k}"] = recall_at_k(retrieved_ids, exp_found, k)
        else:
            out[f"found_recall@{k}"] = None
    return out


def aggregate(per_question: list[dict]) -> dict:
    """Mean metrics across questions, ignoring None for foundational metrics."""
    keys = {k for q in per_question for k in q if k != "retrieved"}
    out = {}
    for k in keys:
        values = [q[k] for q in per_question if q.get(k) is not None]
        out[k] = statistics.mean(values) if values else float("nan")
    return out


def parse_weights(spec: str) -> HybridWeights:
    """Parse 'alpha=0.7,beta=0.2,gamma=0.2,delta=0.4' into HybridWeights."""
    w = HybridWeights()
    if not spec:
        return w
    for part in spec.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        if hasattr(w, k):
            setattr(w, k, float(v))
    return w


def build_retrievers(cfg: dict, weights_override: HybridWeights | None = None):
    graph = CitationGraph.load(cfg["citation_graph"]["path"])
    embedder = Embedder(
        cfg["embeddings"]["model"],
        normalize=cfg["embeddings"].get("normalize", True),
    )
    vstore = VectorStore(
        cfg["vector_store"]["path"],
        cfg["vector_store"]["collection"],
        distance=cfg["vector_store"].get("distance", "cosine"),
    )

    # BM25 needs Paper-like records.
    from arxiv_companion.ingest import Paper as _Paper
    papers = []
    for nid, attrs in graph.g.nodes(data=True):
        if attrs.get("in_corpus"):
            papers.append(_Paper(
                arxiv_id=nid,
                title=attrs.get("title", ""),
                abstract="",
                authors=attrs.get("authors", []) or [],
                year=attrs.get("year"),
            ))

    w = cfg["retrieval"]["weights"]
    hybrid_weights = weights_override or HybridWeights(
        alpha=w["alpha"], beta=w["beta"], gamma=w["gamma"], delta=w["delta"],
    )

    return {
        "vector_only": VectorRetriever(vstore, embedder, graph=graph),
        "bm25": BM25Retriever(papers, graph=graph),
        "hybrid": HybridRetriever(
            vstore=vstore, graph=graph, embedder=embedder,
            weights=hybrid_weights,
            max_hops=cfg["citation_graph"].get("max_hops", 2),
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path,
                    default=Path(__file__).parent / "golden_set.yaml")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--topic", type=str, default=None,
                    help="Restrict to one topic from the golden set.")
    ap.add_argument("--weights", type=str, default="",
                    help="Override hybrid weights, e.g. 'alpha=0.7,beta=0.2'.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional path to write detailed JSON results.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    golden = load_golden(args.golden)
    if args.topic:
        golden = [q for q in golden if q.get("topic") == args.topic]
        if not golden:
            raise SystemExit(f"No questions with topic={args.topic!r}")

    weights = parse_weights(args.weights) if args.weights else None
    retrievers = build_retrievers(cfg, weights_override=weights)

    # Run every retriever on every question.
    per_retriever: dict[str, list[dict]] = defaultdict(list)
    detailed = {}
    k_max = max(K_VALUES)

    for q in golden:
        question = q["question"]
        expected = q["expected"]
        detailed[question] = {"expected": expected, "runs": {}}
        for name, retr in retrievers.items():
            res = evaluate_one(retr, question, expected, k_max=k_max)
            per_retriever[name].append(res)
            detailed[question]["runs"][name] = res

    # Aggregate + print.
    agg = {name: aggregate(rs) for name, rs in per_retriever.items()}
    _print_table(agg)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"aggregate": agg, "detailed": detailed}, f, indent=2)
        print(f"\nWrote detailed results to {args.out}")


def _print_table(agg: dict[str, dict]):
    cols = ["hit@5", "hit@10", "recall@10", "MRR", "AP", "found_recall@10"]
    col_keys = ["hit@5", "hit@10", "recall@10", "mrr", "ap", "found_recall@10"]
    header = f"{'retriever':<18}" + "".join(f"{c:>16}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in agg.items():
        row = f"{name:<18}" + "".join(
            f"{m.get(k, float('nan')):>16.3f}" for k in col_keys
        )
        print(row)


if __name__ == "__main__":
    main()
