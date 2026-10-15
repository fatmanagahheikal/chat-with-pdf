from build_index_pdf import embedder, db
from golden_dataset import GOLDEN_DATASET

def retrieve_chunks(collection, question, top_k=3, section="body"):
    where = {"section": section} if section else None
    q = embedder.encode(question, normalize_embeddings=True)
    results = collection.query(query_embeddings=[q.tolist()], n_results=top_k, where=where)
    pages = [m["page"] for m in results["metadatas"][0]]
    texts = results["documents"][0]
    return pages, texts

def evaluate(collection, top_k=3, verbose=True, dataset=GOLDEN_DATASET, section="body", retrieve_fn=None):
    page_hits = 0
    answer_hits = 0
    for item in dataset:
        if retrieve_fn:
            pages, texts = retrieve_fn(item["question"], top_k)
        else:
            pages, texts = retrieve_chunks(collection, item["question"], top_k, section)
        page_hit = any(p in item["pages"] for p in pages)
        answer_hit = any(item["answer"].lower() in t.lower() for t in texts)
        page_hits += page_hit
        answer_hits += answer_hit
        if verbose:
            mark = "HIT " if answer_hit else "MISS"
            page_mark = "OK" if page_hit else "X "
            print(f"{mark} | page {page_mark} | expected {item['pages']} got {pages} | {item['question']}")
    n = len(dataset)
    if verbose:
        print(f"\nPage Hit Rate@{top_k}:   {page_hits}/{n} = {page_hits / n:.0%}")
        print(f"Answer Hit Rate@{top_k}: {answer_hits}/{n} = {answer_hits / n:.0%}")
    return answer_hits / n

if __name__ == "__main__":
    collection = db.get_collection("clinic_pdf")
    evaluate(collection, top_k=3)