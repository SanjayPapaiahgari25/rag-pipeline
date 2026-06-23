# rag-pipeline

A document question-answering system built from scratch in a single Python file.
It loads a folder of text documents, retrieves the passages most relevant to a
question, and uses a language model to answer **only from those passages** — citing
its sources and refusing when the answer isn't there.

No RAG framework (no LangChain, no LlamaIndex). Chunking, embedding, similarity
search, and the retrieval loop are all implemented directly, so every part of the
pipeline is visible and inspectable.

---

## What it does

Given a question, the system:

1. Finds the 3 document chunks most semantically similar to the question.
2. Passes those chunks to a language model as the only allowed context.
3. Returns an answer with inline citations to the chunks it used.
4. Returns `Not found in the provided documents.` when no chunk supports an answer.

The refusal behavior is the point: the model is constrained to the retrieved
evidence and is not allowed to fall back on its own training knowledge.

---

## How it works

```
docs/*.txt
    │  load_documents()
    ▼
raw text
    │  chunk_text()          fixed-size, 500 chars, 50 overlap
    ▼
chunks
    │  embed_chunks()        all-MiniLM-L6-v2  (local, 384-dim)
    ▼
embeddings  (N × 384)
    │
    ▼
[ query ] ──embed──► query vector (384)
    │  cosine_similarity vs every chunk     (hand-rolled, numpy)
    ▼
top-3 chunks
    │  gpt-4o-mini           answer constrained to context
    ▼
cited answer  (or refusal)
```

**Ingestion** runs once at startup: documents are loaded, split into overlapping
fixed-size chunks, and embedded locally with a Sentence-Transformers model. Nothing
leaves the machine in this phase.

**Query time** runs per question: the question is embedded with the *same* model,
scored against every chunk with cosine similarity, and the top 3 chunks are sent to
`gpt-4o-mini` with a system prompt that forbids using outside knowledge.

---

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/SanjayPapaiahgari25/rag-pipeline.git
cd rag-pipeline
uv sync
```

Set your OpenAI API key in a `.env` file at the repo root:

```
OPENAI_API_KEY=sk-...
```

The `.env` file is gitignored and never committed. On first run, the embedding
model (~80 MB) downloads automatically and is cached locally.

---

## Usage

```bash
uv run python rag.py
```

This starts an interactive loop. Type a question, get a cited answer, type `quit`
to exit.

```
Loading documents...
Embedding chunks...
Ready. Loaded 18 docs, 84 chunks.

Ask a question (or 'quit'): What is the difference between NPS Tier I and Tier II?

NPS Tier I is the core retirement account: contributions are locked in until age 60
with only limited partial withdrawals for specific purposes [Chunk 1]. Tier II is an
optional, voluntary account that can only be opened alongside a Tier I account. It has
no lock-in and allows withdrawals at any time, but does not carry the same tax
deductions as Tier I [Chunk 2][Chunk 3].

Ask a question (or 'quit'): How do I make biryani?

Not found in the provided documents.

Ask a question (or 'quit'): quit
```

The `docs/` folder ships with 18 sample documents across machine learning, system
design, and personal finance. Replace them with your own `.txt` files to query a
different corpus.

---

## Project structure

```
rag-pipeline/
├── docs/                  18 .txt source documents
├── notes/                 build notes and test query bank
├── rag.py                 the entire pipeline
├── .env                   API key (not committed)
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Design decisions

- **No framework.** Implementing retrieval by hand makes the trade-offs explicit
  and means there is no hidden behavior to explain in an interview.
- **Local embeddings, hosted generation.** Embedding with `all-MiniLM-L6-v2` runs
  on-device for free; only generation calls a paid API. This keeps cost proportional
  to questions asked, not corpus size.
- **Cosine similarity by hand.** Computed directly with numpy rather than a vector
  library, to keep the similarity step transparent.
- **Constrained generation.** The system prompt forbids outside knowledge and
  mandates a fixed refusal string, which is what makes the citations trustworthy.

---

## Limitations and known issues

This is a v0.1 baseline. The edges are documented deliberately — they define the
next version.

- **No persistence.** Chunks and embeddings live in memory and are recomputed on
  every startup. Fine for a small corpus, wasteful for a large one.
- **Fixed-size character chunking.** Chunks are cut at a fixed character count, so
  they split mid-sentence and ignore document structure. Overlap mitigates but does
  not solve this.
- **Comparison questions retrieve poorly.** When an answer spans two documents,
  neither half always scores high enough to make the top 3, so the answer can come
  back partial. This is the clearest motivation for a reranking step.
- **Linear search.** Similarity is computed against every chunk in a Python loop —
  acceptable for a few hundred chunks, not for thousands.
- **No conversation memory.** Each question is answered independently; follow-up
  questions that rely on prior turns won't work.
- **Generation depends on a network call.** Answers require a reachable OpenAI API
  and incur per-query cost.

---

## Roadmap (v1.0)

- Persist embeddings in a vector store (ChromaDB) with metadata filtering
- An evaluation harness with accuracy and retrieval-hit metrics, run in CI
- A FastAPI service exposing a `/query` endpoint
- A cross-encoder reranker over the retrieved candidates
- Per-query tracing, token, and cost tracking
- A test suite covering chunking, retrieval, and the refusal path