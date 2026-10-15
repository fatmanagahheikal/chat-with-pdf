from build_index_pdf import build_index
from evaluate_retrieval import evaluate
from bm25_search import make_bm25_retriever
from hybrid_search import make_hybrid_retriever
from golden_paper import GOLDEN_PAPER

collection = build_index("paper.pdf")
bm25_retrieve = make_bm25_retriever(collection)
hybrid_retrieve = make_hybrid_retriever(collection, bm25_retrieve)

methods = {"Semantic": None, "BM25": bm25_retrieve, "Hybrid": hybrid_retrieve}

print("\n===== COMPARISON (body only) =====")
for name, fn in methods.items():
    h1 = evaluate(collection, top_k=1, verbose=False, dataset=GOLDEN_PAPER, retrieve_fn=fn)
    h3 = evaluate(collection, top_k=3, verbose=False, dataset=GOLDEN_PAPER, retrieve_fn=fn)
    print(f"{name:8} | Hit@1 = {h1:.0%} | Hit@3 = {h3:.0%}")

build_index("clinic_info.pdf")