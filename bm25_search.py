import re
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def make_bm25_retriever(collection, section="body"):
    data = collection.get(where={"section": section})
    texts = data["documents"]
    pages = [m["page"] for m in data["metadatas"]]
    bm25 = BM25Okapi([tokenize(t) for t in texts])

    def retrieve(question, top_k=3):
        scores = bm25.get_scores(tokenize(question))
        best = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [pages[i] for i in best], [texts[i] for i in best]

    return retrieve