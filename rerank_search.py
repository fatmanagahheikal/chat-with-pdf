from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def make_rerank_retriever(base_retrieve, candidates=20):
    def retrieve(question, top_k=3):
        pages, texts = base_retrieve(question, candidates)
        scores = reranker.predict([(question, t) for t in texts])
        best = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [pages[i] for i in best], [texts[i] for i in best]

    return retrieve