import time
from build_index_pdf import build_index
from evaluate_retrieval import retrieve_chunks
from bm25_search import make_bm25_retriever
from hybrid_search import make_hybrid_retriever
from rerank_search import make_rerank_retriever
from golden_paper import GOLDEN_PAPER

collection = build_index("paper.pdf")
bm25_retrieve = make_bm25_retriever(collection)
hybrid_retrieve = make_hybrid_retriever(collection, bm25_retrieve)
rerank_retrieve = make_rerank_retriever(hybrid_retrieve)

def semantic_retrieve(question, top_k=3):
    return retrieve_chunks(collection, question, top_k)

methods = {"Semantic": semantic_retrieve, "BM25": bm25_retrieve, "Hybrid": hybrid_retrieve, "Rerank": rerank_retrieve}
questions = [item["question"] for item in GOLDEN_PAPER]

for name, fn in methods.items():
    fn(questions[0])
    start = time.perf_counter()
    for q in questions:
        fn(q)
    total = time.perf_counter() - start
    print(f"{name:8} | {total / len(questions) * 1000:.0f} ms per question")

build_index("clinic_info.pdf")