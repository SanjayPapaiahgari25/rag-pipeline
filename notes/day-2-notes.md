# Day 2 (Saturday) — Revision Notes
**RAG Pipeline — retrieval + generation**

---

## 1. Retrieval — how it works

Three steps at query time:
1. Embed the question with the **same model** (`all-MiniLM-L6-v2`) — same vector space, or scores are meaningless.
2. Compute cosine similarity between the question vector and every chunk vector.
3. Sort descending, return top-k (k=3 for now).

Brute-force search over 24 chunks is correct at this scale — microseconds. Vector databases (FAISS, ChromaDB) only pay off at millions of chunks.

---

## 2. Cosine similarity — the math

```
cos_sim(A, B) = (A · B) / (|A| × |B|)
```

- `np.dot(a, b)` — multiply corresponding elements, sum.
- `np.linalg.norm(a)` — vector length (√ sum of squares).
- Dividing by lengths removes magnitude → only **direction** (meaning) is compared.
- Range −1 to 1: ~1 same meaning, ~0 unrelated.

Worked example:
```
A = [1, 2, 0]   B = [2, 4, 0]   (B is A scaled ×2 — same direction)
dot = 10, |A| = √5, |B| = √20
cos_sim = 10 / (√5 × √20) = 10 / 10 = 1.0   ← magnitude removed, direction preserved
```

Self-test: `cosine_similarity(v, v)` must return ~1.0000 for any vector. Run this before the full test to catch math bugs early.

---

## 3. Top-k selection

```python
top_idx = np.argsort(scores)[::-1][:top_k]   # ascending → reverse → take k
```

You need **indices**, not sorted scores, because the index links back to `all_chunks[i]` — the text and source filename.

---

## 4. What realistic scores look like

- Genuinely relevant chunk: **0.5–0.75**
- Unrelated content: **below ~0.3**
- Score of 0.99 = chunk is nearly word-for-word the question — suspicious

What matters is the **gap** between relevant and irrelevant, not the absolute number.

Today's results confirmed this:
```
Multi-head attention     → transformers.txt    0.664  ✓
Smallest K8s unit        → kubernetes.txt      0.655  ✓
Tier I vs Tier II        → nps_pension.txt     0.370  ✓ (low but correct — see section 7)
Best time for Andamans   → andaman_travel.txt  0.528  ✓
How to make biryani      → best score 0.120    ✓ trap question, correctly low
```

---

## 5. Generation — how it works

LLMs are excellent at reading provided text and synthesizing answers but hallucinate on things outside their knowledge. RAG splits the job:
- **Retrieval** fetches facts from YOUR documents.
- **LLM** only reads and writes — it is the language engine, not the knowledge store.

### Prompt structure

```
System: Answer ONLY from context. Cite [Chunk N]. Say "Not found in the provided documents." if missing.

User:
[Chunk 1 — nps_pension.txt]
<text>

[Chunk 2 — nps_pension.txt]
<text>

Question: <query>
```

### OpenAI call pattern

```python
client = OpenAI()   # reads OPENAI_API_KEY from environment automatically
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ],
)
text = resp.choices[0].message.content
```

---

## 6. Why "ONLY" is the most important word (check 12)

Without it, the model blends retrieved context with training knowledge — answers questions "correctly" using information that didn't come from your documents. Citations become decorative. You lose the guarantee that answers are grounded in your data.

**Failure mode "ONLY" prevents:** confident answers wearing a citation costume that didn't come from your documents.

**The trap test** verifies it: "How do I make biryani?" → must return "Not found in the provided documents." Not a recipe. ✓ Passed today.

---

## 7. The Tier I/II retrieval problem — a real production lesson

Score was 0.370 vs 0.65+ for other questions. Three things happened simultaneously:

**Vocabulary mismatch:** Query used "difference" — the doc never uses that word. It just describes each tier separately.

**Information split across chunks:** The answer to a comparison question requires two passages. No single chunk contains both tiers side by side, so no single chunk scores high.

**Fixed-size chunking ignores topic boundaries:** 500-char cuts don't know whether they're mid-paragraph or mid-idea.

### Fixes in order of complexity

| Fix | What it does | When |
|-----|-------------|------|
| Larger chunk size (1000 chars) | Both tier descriptions fit in one chunk | Trivial, try now if curious |
| Paragraph-aware chunking | Splits on `\n\n` first, then falls back to characters | Week 3 |
| Hybrid search (BM25 + semantic) | Keyword match finds "Tier I"/"Tier II" exactly | Week 4 |
| Reranker (cross-encoder) | Reads query + chunk together, much more accurate | Week 4–5 |
| HyDE | Embeds a hypothetical answer instead of the raw question | Advanced |

**Interview one-liner:** *"Low scores on comparison questions are a chunking granularity problem — the answer spans multiple chunks so no single chunk scores high. Fix: paragraph-aware splitting or a reranker."*

---

## 8. .env pattern

```
# .env (never committed)
OPENAI_API_KEY=sk-...
```

```python
from dotenv import load_dotenv
load_dotenv()   # call BEFORE creating OpenAI client — top of file
```

OpenAI client picks up the key from environment automatically. Key string never appears in code. `git status` must not show `.env`.

---

## 9. Bugs caught in code review (before running)

1. `retrieve()` had no return statement — function returned `None` silently.
2. `embed_chunks(chunk_text(query))` returned a 2D array `(1, 384)` instead of a 1D vector `(384,)` — correct: `model.encode(query)` directly.
3. Scores appended as lists `[source, text, score]` but test loop accessed dict keys `r['score']` — use dicts consistently.
4. Test loop was outside `if __name__ == "__main__":` block — indentation error.

Pattern: bugs 1 and 2 would have produced silent wrong output, not crashes. Silent failures are harder to catch than errors — always add the self-test.

---

## 10. Complete pipeline after today

```
INGESTION (once):
  load_documents() → chunk_text() → embed_chunks() → store in memory

QUERY TIME (per question):
  model.encode(query)              → query vector (384,)
  cosine_similarity × all chunks   → scores list
  sort + top-k                     → retrieved chunks
  build context string             → numbered chunks with sources
  OpenAI gpt-4o-mini               → cited answer
```

---

## 11. Current repo state

- Branch: `main`
- Merged PRs: `feat/retrieval`, `feat/generation`
- Working: full RAG pipeline — ingest → retrieve → answer with citations → refuse correctly

## 12. Tomorrow (Sunday)

- `git checkout -b feat/cli-and-docs`
- Replace the test block with a real CLI loop: `python rag.py` → type questions → get answers → `quit` to exit.
- Load real ~20 documents you actually care about (replace the 4 test docs).
- Basic error handling: empty query, missing docs folder, API failure.
- Write the README: what it does, how to install, how to run, one example Q&A.
- Tag the release: `git tag v0.1`
- That's a shippable portfolio project.