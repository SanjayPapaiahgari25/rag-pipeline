import os
import sys
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
model = None

def cosine_similarity(a, b) -> float:
    # by hand with numpy: np.dot and np.linalg.norm
    # do NOT import a library version
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query: str, chunks: list[dict], embeddings,
             top_k: int = 3) -> list[dict]:
    # embed the query with the SAME model
    # score against every chunk embedding
    # return top_k as [{"text", "source", "score"}], best first
    query_emb = embed_chunks(query)

    scores = []
    for i,embedding in enumerate(embeddings):
        scores.append({
            "source": chunks[i]["source"],
            "text":   chunks[i]["text"],
            "score":  float(cosine_similarity(query_emb, embedding))
        })

    scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return scores[:top_k]


def get_sentence_transformer():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def load_documents(folder: str) -> list[dict]:
    # read all .txt files in folder
    # return [{"source": filename, "text": file_contents}]
    docs = []
    files = sorted(os.listdir(folder))
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text = f.read()
                docs.append({"source": file, "text": text})
    return docs

def chunk_text(text: str, chunk_size: int = 500,
               overlap: int = 50) -> list[str]:
    # fixed-size character chunks with overlap
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start: start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks: list[str]) -> list:
    model = get_sentence_transformer()
    return model.encode(chunks)

def answer(query: str, retrieved: list[dict]) -> str:
    context = "\n\n".join(
        f"[Chunk {i+1} — {r['source']}]\n{r['text']}"
        for i, r in enumerate(retrieved)
    )
    system = (
        "Answer the user's question using ONLY the context below. "
        "Cite chunk numbers like [Chunk 2]. If the answer is not "
        "in the context, say exactly: "
        "'Not found in the provided documents.'"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    # Guard 1 — fail fast if the API key is missing, before loading any models
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    # Guard 2 — docs folder must exist and contain something
    if not os.path.isdir("docs"):
        print("Error: 'docs' folder not found.")
        sys.exit(1)

    documents = load_documents("docs")
    if not documents:
        print("Error: no .txt files found in 'docs'.")
        sys.exit(1)

    print("Loading documents...")
    all_chunks = []
    for doc in documents:
        for piece in chunk_text(doc["text"]):
            all_chunks.append({"source": doc["source"], "text": piece})

    print("Embedding chunks...")
    embeddings = embed_chunks([c["text"] for c in all_chunks])
    print(f"Ready. Loaded {len(documents)} docs, {len(all_chunks)} chunks.\n")

    while True:
        query = input("Ask a question (or 'quit'): ").strip()
        if query.lower() == "quit":
            break
        if not query:
            print("Please enter a question.\n")
            continue

        retrieved = retrieve(query, all_chunks, embeddings)
        # Guard 3 — a failed API call shouldn't kill the whole session
        try:
            print("\n" + answer(query, retrieved) + "\n")
        except Exception as e:
            print(f"\nGeneration failed: {e}\n")