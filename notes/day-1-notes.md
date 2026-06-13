# Day 1 (Friday) — Revision Notes
**RAG Pipeline v0.0 — load, chunk, embed**

---

## 1. Embeddings

- An embedding is a vector (list of numbers) representing the **meaning** of text.
- Model used: `all-MiniLM-L6-v2` → every text becomes **384 numbers**.
- Core property: similar meaning → vectors pointing in similar directions.
  - "How do I reset my password?" ≈ "I forgot my login credentials" (no shared words, close vectors).
- The model learned this from millions of sentence pairs: similar sentences pushed together, dissimilar pushed apart.
- Nobody knows what individual dimensions mean — only the distance property matters.

## 2. Cosine similarity

```
cos_sim(A, B) = (A · B) / (|A| × |B|)
```

- Dot product = multiply corresponding elements, sum them.
- Dividing by lengths removes magnitude → only the **angle** (direction = meaning) is compared.
- Range −1 to 1: ~1 same meaning, ~0 unrelated.
- Implementing by hand tomorrow (numpy: `np.dot`, `np.linalg.norm`).

## 3. Chunking

- Why not embed whole documents:
  1. One vector for a 10-page doc averages topics into mush — matches nothing well.
  2. At answer time you paste **focused passages** into the LLM prompt, not whole files.
- Strategy used: **fixed-size with overlap** (500 chars, 50 overlap).
- Trade-off: smaller chunks = precise retrieval, less context; bigger = more context, noisier matching.

## 4. Why overlap (check 9 — the sharp version)

- Without overlap, an answer-bearing sentence can be **split across a chunk boundary**.
- Each half embeds to a weak, partial meaning → neither half scores high on similarity → retrieval misses the answer even though it's in the docs.
- Overlap gives boundary sentences a second chance to appear **whole** in one chunk.
- One-liner: *"Overlap prevents answer-bearing sentences from being split across chunk boundaries, which would weaken their embeddings and cause retrieval to miss them."*
- Failure location: **retrieval step**, not the LLM.

## 5. Same model for chunks and queries (check 10 — the sharp version)

- Each embedding model learns its **own vector space**; dimensions have no shared meaning across models.
- Cross-model cosine similarity multiplies unrelated concepts → produces a number that means **nothing** (not "slightly worse" — meaningless).
- Danger: if dimensions match (e.g., both 384), nothing crashes — retrieval **silently returns garbage**.
- One-liner: *"Embeddings are only comparable within the same model's vector space."*
- Mental model: two models = two different maps; comparing coordinates across maps never works.

## 6. Overlap arithmetic (check 11 — passed)

- `start += chunk_size - overlap` → with 500/50: chunk 0 = [0, 500), chunk 1 = [450, 950).
- Characters 450–499 appear in both chunks — that's the overlap.
- Verified empirically: last 50 chars of chunk 0 == first 50 chars of chunk 1.

## 7. Code patterns worth remembering

- **Lazy singleton for the model** — load once, reuse; deferred import avoids paying the import cost when not needed.
- **`sorted(os.listdir(...))`** — listdir order is filesystem-dependent; sorting makes chunk indices reproducible.
- **`encoding="utf-8"` in `open()`** — platform default encodings break on curly quotes/em-dashes.
- **Guard clause**: `overlap >= chunk_size` → infinite loop; raise ValueError early.
- Stdlib imports at top of file; only expensive imports deferred.
- Model loads from `~/.cache/huggingface/` each run (into RAM) — it is NOT re-downloading; the HF warning is a cosmetic version-check ping.

## 8. Git / setup learnings

- `git init` creates `master`; GitHub standard is `main` → rename with `git branch -M main`.
- "src refspec main does not match any" = branch doesn't exist locally.
- "'origin' does not appear to be a git repository" = remote never added (`git remote add origin <url>`).
- "Repository not found" on HTTPS = repo doesn't exist, username wrong, or private + unauthenticated.
- SSH auth: generate key (`ssh-keygen -t ed25519`), add public key to GitHub, verify with `ssh -T git@github.com`.
- Feature branch flow from tomorrow: `git checkout -b feat/<name>` → commit → push → PR → merge → pull main.
- Commit/branch convention: `feat:`, `fix:`, `docs:` (Conventional Commits).
- uv files: commit `.python-version`, `uv.lock`, `pyproject.toml`. Ignore `.venv/`, `__pycache__/`, `.env`.

## 9. Current state

- Repo: `rag-pipeline` on GitHub, branch `main`, commit `f2ee475`.
- Working: `load_documents()` → `chunk_text()` → `embed_chunks()`.
- 4 docs → 24 chunks → embeddings shape (24, 384). All rubric checks passed.

## 10. Tomorrow (Saturday)

- First command: `git checkout -b feat/retrieval`
- Morning: `cosine_similarity()` by hand (numpy) → `retrieve(query, chunks, embeddings, top_k=3)` → 5-question sanity test (each question should hit exactly one doc + one trap question).
- Afternoon: `.env` + python-dotenv setup, `answer()` with gpt-4o-mini, test the "Not found in the provided documents" case deliberately.