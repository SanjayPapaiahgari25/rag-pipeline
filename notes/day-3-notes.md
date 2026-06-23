# Day 3 (Sunday) — Revision Notes

**RAG Pipeline v0.1 — CLI, real corpus, README, ship it**

Day 3 had almost no new theory. The learning was *engineering judgment*: what makes
a project shippable, what error handling actually matters, and what an interviewer
reads in a repo.

---

## 1. What exists now

A complete, runnable portfolio project. `uv run python rag.py` starts an interactive
loop over a real 18-document corpus, answers with citations, refuses out-of-scope
questions, and exits cleanly. Tagged `v0.1`.

---

## 2. The CLI loop + three guards

The `__main__` block changed from rubric tests to an interactive loop. Three guards,
ordered deliberately:

1. **Missing API key — checked FIRST**, before loading the embedding model. Failing
   fast matters: don't make the user wait ~10s to embed chunks only to crash on a
   missing key.
2. **Missing / empty docs folder** — exit with a clear message instead of an
   obscure traceback.
3. **Per-query API failure** — wrapped in try/except *inside* the loop, so one bad
   call re-prompts instead of killing the whole session.

Lesson: error handling isn't "handle everything." It's handling the few failures
that are *likely* and would otherwise produce a confusing crash. Anything more is
over-engineering for v0.1.

---

## 3. The corpus

18 real `.txt` files (8 ML/AI, 6 system design, 4 personal finance). The point of a
real corpus isn't volume — it's that **near-neighbor documents make retrieval
non-trivial**. `kv_cache.txt` and `speculative_decoding.txt` both discuss inference
speedups; `index_funds.txt` and `mutual_funds.txt` overlap heavily. If every doc were
on a unrelated topic, top-k would always be trivially correct and prove nothing.

---

## 4. How citations actually work

The `[Chunk N]` labels are generated in `answer()`:

```python
f"[Chunk {i+1} — {r['source']}]\n{r['text']}"
```

- `[Chunk 1]` = the **highest-scoring retrieved chunk for THIS query**, `[Chunk 2]`
  the second, and so on. They are positions in the retrieval result, best-first —
  **not stable identifiers**. Chunk 1 for a different question is different text.
- **Key gap:** the model emits a bare *number*. The source filename lives only in the
  context string we feed it — it never reaches the output. So a reader of the answer
  can't tell which file `[Chunk 1]` came from. For a system whose whole value is
  *verifiable* citations, that's a real weakness.
- **Minimal fix:** make the citation carry the filename — cite `[nps.txt]` instead of
  `[Chunk 1]`. Two-line change to the context format and the system prompt.

**Why citations clustered at the end:** the prompt specified the citation *format* but
not its *placement*. When the model fuses three chunks into smooth prose where no
single sentence maps to one chunk, it hedges by listing all sources at the end.
Placement is non-deterministic unless the prompt pins it. Trailing citations are a
legitimate style; inline is a polish item, not a correctness one.

---

## 5. The README — the Limitations section is the point

Five sections plus a roadmap. The section that does the real work is **Limitations
and known issues**. Most self-taught RAG repos stop at "it works." Naming six
specific failure modes — no persistence, fixed-size chunking, weak comparison-question
retrieval, linear search, no memory, network dependence — signals you understand the
system's *edges*, not just its happy path. Each limitation maps to a roadmap item, so
"what would you improve?" answers itself.

Honesty check: the README example must match what the program actually prints (real
chunk count, real citation format). A README that disagrees with the code on line one
is a red flag.

---

## 6. Shipping = the v0.1 tag

```bash
git tag -a v0.1 -m "Working RAG pipeline with CLI, citations, and refusal"
git push origin v0.1
```

The tag is the line between "I'm building a project" and "here's the repo, clone and
run it." A Releases entry an interviewer can click signals you think in versions.

---

## 7. Current repo state

```
rag-pipeline/
├── docs/            18 real .txt files
├── notes/           day1, day2, day3 notes + test_queries.md
├── rag.py           ~90 lines, production CLI, 3 guards
├── .env             not committed
├── pyproject.toml
├── uv.lock
└── README.md        5 sections + limitations + roadmap
```

Four+ merged PRs. One `v0.1` tag. A repo that clones and runs in under two minutes.

---

## 8. The one failure mode to remember

Comparison questions whose answer spans two documents retrieve poorly — neither half
always scores into the top 3. This isn't a bug to patch now; it's the **concrete
justification for a cross-encoder reranker** in v1.0. Knowing *why* your system fails
and what fixes it is a stronger interview answer than a system with no known flaws.

---

## 9. Next: v1.0 hardening

Start with the **eval harness** — reuse the rule-based `must_include` checker, grow it
to ~30 cases with accuracy + retrieval-hit metrics. Building it first means every later
upgrade (ChromaDB, FastAPI, reranker) has a before/after number to prove it helped.

Order: eval harness → tests → ChromaDB → FastAPI → observability/cost → reranker →
deploy + tag `v1.0`.