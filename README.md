# GraphRAG-RL

Context selection for question answering over Uzbek legislation (Oliy Majlis Senate archive) with a legal knowledge graph and a reinforcement-learned agent.

For every question the system decides which fragments of the corpus go into the LLM context, and how many: from 1 up to `K_max`.

- **Router** (logistic regression on bge-m3 question vectors) sends a question to one of three branches:
  - regular questions → **RL ranker** over hybrid (BM25 + dense + cross-encoder) candidates;
  - structural "count" questions (how many articles / parts / points, Q1 and Q1x) → **struct agent**: a graph policy trained with SFT on expert trajectories, then GRPO; it walks the knowledge graph and the document structure;
  - "version" questions (what a law amended, Q2) → a deterministic rule over the AMENDS edges of the graph.
- **Reward** of the agent: R = 0.5·F1 + 0.2·coverage + 0.2·diversity + 0.1·coherence.

## Results

Test split, 212 questions, retrieval budget of up to 12 fragments per question.

| method | recall@12 (± SE) | mean context tokens per question |
|---|---|---|
| **GraphRAG-RL** | **0.890 ± 0.021** | **1953** |
| [PropRAG](https://github.com/ReLink-Inc/PropRAG) ([paper](https://arxiv.org/abs/2504.18070)) | 0.635 ± 0.033 | 3372 |
| Naive RAG (dense top-12) | 0.609 ± 0.033 | 3304 |
| [HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG) | 0.604 ± 0.033 | 3336 |

## Repository layout

| path | content |
|---|---|
| `graphrag/` | corpus ingest, knowledge graph construction (M1–M2), stores (PostgreSQL, Qdrant, Neo4j), retrieval |
| `grl/` | environment, models (graph policy, ranker), SFT / GRPO training, evaluation |
| `scripts/` | training and evaluation entry points (`train.py`, `m7_*.py`), DGX Spark helper |
| `config/` | configuration (`train.yaml`: environment, reward, model, training) |
| `tests/` | unit and integration tests |
| `checkpoints/` | the frozen system: router, ranker, agent seeds ([details](checkpoints)) |
| `baselines/` | baseline methods: links to the official code and our runner scripts |

## Data

The corpus, the knowledge graph, the question sets and the indexes are not published yet. The code expects them under `data/` (see `config/` and `.env.example` for the stores).

## Setup

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in the store addresses and OPENAI_API_KEY
docker compose up -d   # PostgreSQL, Qdrant, Neo4j for development
```
