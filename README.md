# arXiv Research Companion

> Hybrid RAG over arXiv papers that combines **dense vector retrieval** with **citation-graph traversal**. Pure semantic search misses foundational work; pure citation following misses recent and topically adjacent papers. This project does both, and lets you measure the difference.

When you ask *"What are the foundational papers behind diffusion models?"*, a vanilla RAG system finds papers that *talk about* diffusion. This one also walks back through the citation network from those semantic hits and surfaces the actual ancestors — Sohl-Dickstein 2015, DDPM, score matching — even when they don't appear in the top semantic results.

---

## Why this is different from "chat with your PDFs"

Most RAG demos treat retrieval as one problem: nearest neighbors in embedding space. That misses how scientific knowledge is actually structured. Two papers about the same topic can use different vocabulary; a foundational paper might be linguistically distant from its descendants; influence flows along citation edges, not embedding similarity.

This project models that explicitly:

- **Vector index** over paper sections (abstract, intro, method) for semantic recall.
- **Citation graph** built from Semantic Scholar references, with PageRank precomputed for influence scoring.
- **Hybrid retriever** that takes semantic hits as seeds, walks the graph for ancestors/descendants, then fuses by a tunable weighting of semantic score and graph centrality.
- **Evaluation harness** with a curated golden set comparing the hybrid approach to vector-only and BM25 baselines, with hit-rate, MRR, and recall@k.

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
   top-k ──────────▶│  build cited prompt ──▶ LLM ──▶ answer with │──▶ response
                    │                                 arxiv links │
                    └─────────────────────────────────────────────┘
```

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure (optional — defaults work with local embeddings + Anthropic API)
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# 3. Build the index for a topic
python scripts/build_index.py --query "diffusion models" --max-papers 200

# 4. Ask a question
python scripts/demo_query.py "What are the foundational papers behind diffusion models?"

# 5. Run the eval harness comparing hybrid vs vector-only
python scripts/run_eval.py
```

---

## How the hybrid retriever works

The retriever takes a query and runs three passes:

1. **Vector seeds.** Embed the query, retrieve the top-`k_seed` chunks from ChromaDB, deduplicate to paper-level. These are the "semantically relevant" hits.

2. **Graph expansion.** For each seed, walk the citation graph up to `max_hops` (default 2):
   - **Ancestors** (papers the seeds cite, transitively) → likely foundational.
   - **Descendants** (papers that cite the seeds) → recent extensions and applications.
   - **Siblings** (papers that share citations with seeds) → topical neighbors that semantic search may have missed.

3. **Fused reranking.** Combine candidates with:
   ```
   score(p) = α · semantic_sim(q, p)         # how relevant is the paper itself
            + β · pagerank(p)                # how influential is it in the graph
            + γ · seed_proximity(p)          # how close to a semantic hit
            - δ · hop_penalty(p)             # decay for graph-only papers
   ```
   Weights `α, β, γ, δ` live in `config.yaml`; the eval harness sweeps them.

The intuition: a paper that scores high on semantic similarity *and* sits at a high-centrality node in the citation neighborhood is exactly what you want for "foundational." Pure vector ranking misses the centrality signal; pure citation ranking misses query relevance.

---

## Evaluation

The golden set lives in `eval/golden_set.yaml`. Each question lists expected arXiv IDs (papers an expert would say *should* appear in the top results), tagged by type:

- `foundational` — must include ancestor/seminal work
- `recent` — must include 2023+ extensions
- `survey` — broad coverage across subareas
- `specific` — a precise method or result

Metrics computed by `eval/evaluate.py`:

- **Hit Rate @ k** — fraction of questions where ≥1 expected paper appears in top-k
- **Recall @ k** — fraction of expected papers retrieved
- **MRR** — mean reciprocal rank of the first expected paper
- **Foundational recall** — recall restricted to expected papers tagged `foundational` (the metric pure-vector RAG fails on)

Example output:

```
                       hit@5   hit@10   recall@10   MRR    found_recall@10
vector_only            0.62    0.78      0.41       0.51       0.28
bm25                   0.51    0.69      0.34       0.42       0.19
hybrid (α=.6,β=.3)     0.74    0.89      0.58       0.66       0.61
```

The headline result is `foundational_recall@10`: hybrid more than doubles it because ancestor papers come in through the graph walk, not the embeddings.

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
│   ├── retrieval.py             # vector / bm25 / hybrid retrievers
│   ├── generation.py            # LLM answer synthesis (Anthropic/OpenAI/local)
│   └── cli.py                   # entry points
├── eval/
│   ├── golden_set.yaml          # curated questions + expected papers
│   ├── metrics.py               # hit rate, MRR, recall@k
│   └── evaluate.py              # comparison harness
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
  model: sentence-transformers/all-MiniLM-L6-v2   # free, local, 384-dim
  batch_size: 64

vector_store:
  path: ./data/chroma
  collection: arxiv_chunks

citation_graph:
  path: ./data/citations.gpickle
  max_hops: 2
  semantic_scholar_rps: 1.0     # be polite — free tier is rate-limited

retrieval:
  k_seed: 20                    # vector seeds
  k_expand: 30                  # graph candidates per seed
  k_final: 10                   # papers returned to LLM
  weights:
    alpha: 0.6                  # semantic similarity
    beta:  0.3                  # pagerank / influence
    gamma: 0.2                  # seed proximity
    delta: 0.4                  # hop penalty

generation:
  provider: anthropic           # anthropic | openai | none
  model: claude-sonnet-4-6
  max_tokens: 1024
```

---

## Roadmap

- [ ] Cross-encoder reranker before generation (e.g. `ms-marco-MiniLM-L-6-v2`)
- [ ] Section-aware chunking (treat related-work / experiments differently)
- [ ] Graph-attentive retrieval with [PageRank personalized to the query](https://en.wikipedia.org/wiki/PageRank#Personalized_PageRank)
- [ ] Streamlit demo UI with citation graph visualization
- [ ] Incremental indexing (only fetch new papers since last run)
- [ ] Multi-hop QA evaluation (questions that need synthesis across papers)

---

## Acknowledgements

- arXiv API and the [arxiv](https://pypi.org/project/arxiv/) Python wrapper
- [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/) for citation data
- [ChromaDB](https://www.trychroma.com/) for the vector store
- [sentence-transformers](https://www.sbert.net/) for embeddings
- [NetworkX](https://networkx.org/) for the citation graph

## License

MIT — see [LICENSE](./LICENSE).
