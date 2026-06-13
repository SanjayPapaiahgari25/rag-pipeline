import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    # Check 1: runs end-to-end without errors
    documents = load_documents("docs")
    print(f"Loaded {len(documents)} documents")  # expect 4

    all_chunks = []
    for doc in documents:
        for piece in chunk_text(doc["text"]):
            all_chunks.append({"source": doc["source"], "text": piece})

    # Check 2: chunk count > document count
    print(f"Total chunks: {len(all_chunks)}")  # expect ~20-24, definitely > 4

    embeddings = embed_chunks([c["text"] for c in all_chunks])

    # Check 3: embedding dimension is 384
    print(f"Embedding shape: {embeddings.shape}")  # expect (chunk_count, 384)

    # Check 4: source tracking works
    print(f"Chunk 12 came from: {all_chunks[12]['source']}")
    # should print a real filename, e.g. kubernetes.txt

    # Check 6: short document doesn't crash
    tiny = chunk_text("This is a very short document.")
    print(f"Short doc -> {len(tiny)} chunk(s): {tiny}")
    # expect 1 chunk, no empty strings, no crash

    # Check 7: overlap verification
    chunks = chunk_text(documents[0]["text"])
    print("End of chunk 0: ...", chunks[0][-50:])
    print("Start of chunk 1:", chunks[1][:50])
    # these two lines should print THE SAME 50 characters

    # test positive case
    q = "What is the difference between Tier I and Tier II?"
    retrieved = retrieve(q, all_chunks, embeddings)
    print(answer(q, retrieved))

    # test negative case (the trap)
    q = "How do I make biryani?"
    retrieved = retrieve(q, all_chunks, embeddings)
    print(answer(q, retrieved))