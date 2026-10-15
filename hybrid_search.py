from evaluate_retrieval import retrieve_chunks

def make_hybrid_retriever(collection, bm25_retrieve, candidates=20, k=60):
    def retrieve(question, top_k=3):
        sem_pages, sem_texts = retrieve_chunks(collection, question, candidates)
        bm_pages, bm_texts = bm25_retrieve(question, candidates)

        scores = {}
        page_of = {}
        for pages, texts in [(sem_pages, sem_texts), (bm_pages, bm_texts)]:
            for rank, (page, text) in enumerate(zip(pages, texts), start=1):
                scores[text] = scores.get(text, 0) + 1 / (k + rank)
                page_of[text] = page

        best = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [page_of[t] for t in best], best

    return retrieve