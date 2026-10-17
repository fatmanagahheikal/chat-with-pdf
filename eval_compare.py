from build_index_pdf import build_index
from evaluate_retrieval import evaluate
from bm25_search import make_bm25_retriever
from hybrid_search import make_hybrid_retriever
from rerank_search import make_rerank_retriever
from golden_dataset import GOLDEN_DATASET
from golden_paper import GOLDEN_PAPER

tests = [("paper.pdf", GOLDEN_PAPER), ("clinic_info.pdf", GOLDEN_DATASET)]

for pdf_path, dataset in tests:
    collection = build_index(pdf_path)
    bm25_retrieve = make_bm25_retriever(collection)
    hybrid_retrieve = make_hybrid_retriever(collection, bm25_retrieve)
    rerank_retrieve = make_rerank_retriever(hybrid_retrieve)
    methods = {"Semantic": None, "BM25": bm25_retrieve, "Hybrid": hybrid_retrieve, "Rerank": rerank_retrieve}

    print(f"\n===== {pdf_path} =====")
    for name, fn in methods.items():
        h1 = evaluate(collection, top_k=1, verbose=False, dataset=dataset, retrieve_fn=fn)
        h3 = evaluate(collection, top_k=3, verbose=False, dataset=dataset, retrieve_fn=fn)
        print(f"{name:8} | Hit@1 = {h1:.0%} | Hit@3 = {h3:.0%}")