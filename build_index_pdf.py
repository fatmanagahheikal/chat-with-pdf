import chromadb
from sentence_transformers import SentenceTransformer
from pdf_chunks import load_pdf_chunks

# embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
db = chromadb.PersistentClient(path="chroma_db")

def build_index(pdf_path, chunk_size=500, chunk_overlap=150):
    chunks = load_pdf_chunks(pdf_path, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunks from {pdf_path}")

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    try:
        db.delete_collection("clinic_pdf")
    except Exception:
        pass
    collection = db.create_collection(name="clinic_pdf")
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": pdf_path, "page": c["page"], "section": c["section"]} for c in chunks],
    )
    print(f"Saved {collection.count()} chunks to the 'clinic_pdf' collection")
    return collection

if __name__ == "__main__":
    build_index("clinic_info.pdf")