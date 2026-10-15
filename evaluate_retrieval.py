from build_index_pdf import embedder, db
from golden_dataset import GOLDEN_DATASET

def retrieve_chunks(collection, question, top_k=3):
    q = embedder.encode(question, normalize_embeddings=True)
    results = collection.query(query_embeddings=[q.tolist()], n_results=top_k)
    pages = [m["page"] for m in results["metadatas"][0]]
    texts = results["documents"][0]
    return pages, texts

def evaluate(collection, top_k=3, verbose=True):
    page_hits = 0
    answer_hits = 0
    for item in GOLDEN_DATASET:
        pages, texts = retrieve_chunks(collection, item["question"], top_k)
        page_hit = any(p in item["pages"] for p in pages)
        answer_hit = any(item["answer"].lower() in t.lower() for t in texts)
        page_hits += page_hit
        answer_hits += answer_hit
        if verbose:
            mark = "HIT " if answer_hit else "MISS"
            page_mark = "OK" if page_hit else "X "
            print(f"{mark} | page {page_mark} | got {pages} | {item['question']}")
    n = len(GOLDEN_DATASET)
    if verbose:
        print(f"\nPage Hit Rate@{top_k}:   {page_hits}/{n} = {page_hits / n:.0%}")
        print(f"Answer Hit Rate@{top_k}: {answer_hits}/{n} = {answer_hits / n:.0%}")
    return answer_hits / n

if __name__ == "__main__":
    collection = db.get_collection("clinic_pdf")
    evaluate(collection, top_k=3)