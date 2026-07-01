# arXiv Research Companion

A retrieval-augmented generation (RAG) system for arXiv papers. It combines dense vector search with citation graph traversal to answer research questions with cited, grounded responses from Claude.

Ask *"What are the foundational papers behind diffusion models?"* — the system finds semantically relevant papers, walks the citation graph to surface influential ancestor papers that pure vector search misses, and hands the whole set to Claude to write a cited answer.

---

## Why hybrid retrieval

Most RAG demos stop at vector search: embed the query, find nearest neighbors, hand them to an LLM. That misses how scientific knowledge is structured. A foundational paper often uses different vocabulary than modern papers building on it. Sohl-Dickstein et al. 2015 talks about "nonequilibrium thermodynamics" — it wouldn't rank highly for a query like "diffusion models for image generation," but every diffusion paper cites it.

This project models influence explicitly:

- **Vector index over paper abstracts** for semantic recall
- **Citation graph from Semantic Scholar** with PageRank precomputed for influence scoring
- **Hybrid retriever** that takes semantic hits as seeds, walks the graph for ancestors and descendants, then reranks by a tunable combination of similarity and centrality
- **Evaluation harness** comparing the hybrid approach to vector-only and BM25 baselines

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │                    INGEST                   │
                    │                                             │
   arXiv API ──────▶│  fetch papers ──▶ chunk sections ──▶ embed  │──▶ ChromaDB
                    │       │                                     │
                    │       ▼                                     │
   Semantic Scholar ▶  fetch refs/citations ─────────────────────▶│──▶ NetworkX graph
                    │                                             │     (+ PageRank)
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │                  RETRIEVAL                  │
                    │                                             │
   query ──▶ embed ─▶  vector search ──▶ seed papers              │
                    │       │                                     │
                    │       ▼                                     │
                    │  graph walk ──▶ ancestors + descendants     │
                    │       │                                     │
                    │       ▼                                     │
                    │  rerank by α·semantic + β·pagerank + γ·hops │──▶ top-k papers
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │                 GENERATION                  │
                    │                                             │
   top-k ──────────▶│  build cited prompt ──▶ Claude ──▶ answer   │──▶ response
                    │                                 with [1][2] │       with
                    │                                 citations   │    arxiv links
                    └─────────────────────────────────────────────┘
```

---

## What each piece uses

| Component | Tech |
|-----------|------|
| Paper metadata | [arXiv API](https://info.arxiv.org/help/api/index.html) |
| Citation graph edges | [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim) |
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| Graph | [NetworkX](https://networkx.org/) with PageRank |
| Answer generation | [Claude](https://www.anthropic.com/api) (Sonnet 4.6 by default; OpenAI GPT also supported) |
| Baseline comparison | BM25 via `rank-bm25` |

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get API keys and configure
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY (required for generation)
#           add SEMANTIC_SCHOLAR_API_KEY (strongly recommended, see below)

# 3. Build the index for a topic
python scripts/build_index.py --query "diffusion models" --max-papers 200

# 4. Ask a question
python scripts/demo_query.py "What are the foundational papers behind diffusion models?"
```

You'll get a ranked list of papers followed by a Claude-generated essay with bracketed citations pointing back to the retrieved papers.

### About the Semantic Scholar key

The Semantic Scholar API is what supplies the citation edges. Without a key you're on the anonymous free tier, which throttles aggressively — in practice you'll get `HTTP 429` on most requests and your citation graph will be nearly empty (all `via=semantic` results, no `via=ancestor`).

Request a free key at https://www.semanticscholar.org/product/api#api-key-form. Academic emails are prioritized and turnaround is typically same-day. Once you have it:

```
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

You can still run the system without one (vector retrieval and generation both work), but you won't see the graph contribution that makes hybrid retrieval interesting.

### Running without an LLM

If you want to inspect retrieval quality without spending API credits:

```bash
python scripts/demo_query.py --no-llm "your question here"
```

This prints the ranked papers with score breakdowns but skips the Claude call.

---

## How the hybrid retriever works

Three passes:

**1. Vector seeds.** Embed the query, retrieve top-`k_seed` chunks from ChromaDB, deduplicate to paper level.

**2. Graph expansion.** For each seed, walk the citation graph up to `max_hops`:
- *Ancestors* (papers the seeds cite, transitively) — likely foundational
- *Descendants* (papers that cite the seeds) — recent extensions
- *Siblings* (papers sharing citations with seeds) — topical neighbors semantic search may have missed

**3. Fused reranking.** Combine candidates with:

```
score(p) = α · semantic_sim(q, p)         # how relevant is the paper itself
         + β · pagerank(p)                # how influential in the citation graph
         + γ · seed_proximity(p)          # how close to a semantic hit
         - δ · hop_penalty(p)             # decay for graph-only papers
```

Weights `α, β, γ, δ` live in `config.yaml`; the eval harness sweeps them.

The intuition: a paper scoring high on semantic similarity *and* sitting at a high-centrality node in the citation neighborhood is exactly what "foundational" looks like. Vector-only ranking misses centrality; citation-only ranking misses query relevance.

---

## Evaluation

The golden set in `eval/golden_set.yaml` covers 8 topics (diffusion, transformers, RLHF, chain-of-thought, word embeddings, RAG, NAS, contrastive learning), each with expected arXiv IDs tagged `foundational` or `recent`.

Metrics computed by `eval/evaluate.py`:
- **Hit Rate @ k** — fraction of questions where ≥1 expected paper is in top-k
- **Recall @ k** — fraction of expected papers retrieved
- **MRR** — mean reciprocal rank of the first expected paper
- **Foundational recall @ k** — recall restricted to `foundational`-tagged papers (the metric pure-vector RAG fails on)

Run with:
```bash
python scripts/run_eval.py
python scripts/run_eval.py --topic diffusion
python scripts/run_eval.py --weights "alpha=0.7,beta=0.4"
```

The headline metric is `foundational_recall@10` — this is where the citation graph does its work.

---

## Project layout

```
arxiv-research-companion/
├── README.md
├── requirements.txt
├── config.yaml                  # weights, model choices, paths
├── .env.example
├── arxiv_companion/
│   ├── ingest.py                # arXiv + Semantic Scholar fetching
│   ├── store.py                 # ChromaDB + NetworkX persistence
│   ├── retrieval.py             # vector / BM25 / hybrid retrievers
│   ├── generation.py            # LLM answer synthesis (Claude / OpenAI / none)
│   └── cli.py                   # entry points
├── eval/
│   ├── golden_set.yaml
│   ├── metrics.py
│   └── evaluate.py
├── scripts/
│   ├── build_index.py
│   ├── demo_query.py
│   └── run_eval.py
└── tests/
    └── test_retrieval.py
```

---

## Configuration

`config.yaml` controls the moving parts without code changes:

```yaml
embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 64

vector_store:
  path: ./data/chroma
  collection: arxiv_chunks

citation_graph:
  path: ./data/citations.gpickle
  max_hops: 2
  semantic_scholar_rps: 1.0
  pagerank_alpha: 0.85

retrieval:
  k_seed: 20
  k_expand: 30
  k_final: 10
  weights:
    alpha: 0.6     # semantic similarity
    beta:  0.3     # pagerank / influence
    gamma: 0.2     # seed proximity
    delta: 0.4     # hop penalty

generation:
  provider: anthropic           # anthropic | openai | none
  model: claude-sonnet-4-6
  max_tokens: 1024
```

---

## Requirements

- Python 3.10+
- Anthropic API key (or OpenAI, or run with `--no-llm`)
- Semantic Scholar API key (strongly recommended for the citation graph to be useful)
- ~200MB disk space for embeddings model + indexed data

---

## Roadmap

- [ ] Cross-encoder reranker before generation
- [ ] Section-aware chunking (use full paper text, not just abstracts)
- [ ] Personalized PageRank conditioned on the query
- [ ] Streamlit UI with interactive citation graph visualization
- [ ] Incremental indexing (only fetch papers new since last run)
- [ ] Multi-hop QA evaluation

---

## Acknowledgements

- [arXiv](https://arxiv.org/) for the paper corpus and the [arxiv](https://pypi.org/project/arxiv/) Python client
- [Semantic Scholar](https://www.semanticscholar.org/) for the citation graph data
- [Anthropic Claude](https://www.anthropic.com/) for answer generation
- [ChromaDB](https://www.trychroma.com/), [sentence-transformers](https://www.sbert.net/), [NetworkX](https://networkx.org/)

## License

MIT — see [LICENSE](./LICENSE).
