import chromadb
from sentence_transformers import SentenceTransformer
from pdf_chunks import load_pdf_chunks

PDF_PATH = "clinic_info.pdf"

chunks = load_pdf_chunks(PDF_PATH)
print(f"Created {len(chunks)} chunks from {PDF_PATH}")

texts = [c["text"] for c in chunks]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, normalize_embeddings=True)

db = chromadb.PersistentClient(path="chroma_db")
try:
    db.delete_collection("clinic_pdf")
except Exception:
    pass
collection = db.create_collection(name="clinic_pdf")
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[{"source": PDF_PATH, "page": c["page"]} for c in chunks],
)
print(f"Saved {collection.count()} chunks to the 'clinic_pdf' collection")