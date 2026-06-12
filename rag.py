import os

model = None

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