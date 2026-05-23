"""Answer synthesis on top of retrieved papers.

Supports multiple LLM backends so the project runs whether you have an
Anthropic key, an OpenAI key, or neither. With `provider: none` you get
the retrieved papers without any generation — useful for eval-only runs.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Sequence

from .retrieval import RetrievedPaper

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an arXiv research assistant. The user asks questions \
about a body of papers. You will be given a numbered list of candidate papers \
that a hybrid vector + citation-graph retriever surfaced as relevant. Your job:

1. Synthesize a clear, accurate answer that draws on the provided papers.
2. Cite every claim with bracketed indices matching the candidate list, e.g. [3].
3. If a paper appears foundational (cited heavily by other candidates, older), \
   say so explicitly. If a paper is a recent extension, say so.
4. Do not invent papers, results, or citations. If the candidates don't support \
   an answer, say what's missing rather than guessing.
5. End with a "Key references" section listing the bracketed numbers you cited \
   with their arXiv IDs."""


@dataclass
class GenerationResult:
    answer: str
    used_papers: list[RetrievedPaper]
    provider: str
    model: str | None


def _format_candidates(papers: Sequence[RetrievedPaper]) -> str:
    lines = []
    for i, p in enumerate(papers, start=1):
        yr = f", {p.year}" if p.year else ""
        marker = ""
        if p.source == "ancestor":
            marker = " [graph: ancestor]"
        elif p.source == "descendant":
            marker = " [graph: descendant]"
        title = p.title or "(title unavailable)"
        lines.append(
            f"[{i}] arXiv:{p.arxiv_id}  {title}{yr}{marker}\n"
            f"     score={p.score:.3f}  {p.explain()}"
        )
    return "\n".join(lines)


def _build_user_prompt(query: str, papers: Sequence[RetrievedPaper]) -> str:
    return (
        f"Question: {query}\n\n"
        f"Candidate papers (ranked by hybrid retriever):\n\n"
        f"{_format_candidates(papers)}\n\n"
        f"Write the answer now. Cite candidates by bracketed index."
    )


# ---------- Providers ----------

class AnthropicGenerator:
    provider = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024, temperature: float = 0.2):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. `pip install anthropic` or "
                "set generation.provider to 'openai' or 'none'."
            ) from e
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str, papers: Sequence[RetrievedPaper]) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(query, papers)}],
        )
        # `content` is a list of blocks; concatenate any text blocks.
        return "".join(getattr(block, "text", "") for block in msg.content)


class OpenAIGenerator:
    provider = "openai"

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1024, temperature: float = 0.2):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. `pip install openai` or "
                "set generation.provider to 'anthropic' or 'none'."
            ) from e
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str, papers: Sequence[RetrievedPaper]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query, papers)},
            ],
        )
        return resp.choices[0].message.content or ""


class NoOpGenerator:
    """Returns a deterministic, LLM-free summary. Useful for eval-only runs
    and for environments without any API keys."""
    provider = "none"
    model: str | None = None

    def generate(self, query: str, papers: Sequence[RetrievedPaper]) -> str:
        lines = [f"Top {len(papers)} candidates for: {query}", ""]
        for i, p in enumerate(papers, start=1):
            yr = f" ({p.year})" if p.year else ""
            lines.append(
                f"[{i}] arXiv:{p.arxiv_id} — {p.title or '(no title)'}{yr}"
            )
            lines.append(f"    {p.explain()}")
        lines.append("")
        lines.append("(No LLM provider configured; showing raw retrieval.)")
        return "\n".join(lines)


def build_generator(config: dict):
    cfg = config.get("generation", {})
    provider = (cfg.get("provider") or "none").lower()
    model = cfg.get("model")
    max_tokens = int(cfg.get("max_tokens", 1024))
    temperature = float(cfg.get("temperature", 0.2))

    if provider == "anthropic":
        return AnthropicGenerator(model=model or "claude-sonnet-4-6",
                                  max_tokens=max_tokens, temperature=temperature)
    if provider == "openai":
        return OpenAIGenerator(model=model or "gpt-4o-mini",
                               max_tokens=max_tokens, temperature=temperature)
    return NoOpGenerator()


def answer(query: str, papers: Sequence[RetrievedPaper], generator) -> GenerationResult:
    text = generator.generate(query, papers)
    return GenerationResult(
        answer=text,
        used_papers=list(papers),
        provider=generator.provider,
        model=getattr(generator, "model", None),
    )
