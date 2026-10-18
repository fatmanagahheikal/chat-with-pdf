from build_index_pdf import build_index
from evaluate_retrieval import evaluate
from bm25_search import make_bm25_retriever
from hybrid_search import make_hybrid_retriever
from rerank_search import make_rerank_retriever
from semantic_chunks import load_semantic_chunks
from llm_chunks import load_llm_chunks
from golden_dataset import GOLDEN_DATASET
from golden_paper import GOLDEN_PAPER

tests = [("paper.pdf", GOLDEN_PAPER), ("clinic_info.pdf", GOLDEN_DATASET)]
chunkers = {"Fixed": None, "Semantic": load_semantic_chunks, "LLM": load_llm_chunks}

for pdf_path, dataset in tests:
    print(f"\n===== {pdf_path} =====")
    for chunk_name, chunker in chunkers.items():
        collection = build_index(pdf_path, chunker=chunker)
        bm25_retrieve = make_bm25_retriever(collection)
        hybrid_retrieve = make_hybrid_retriever(collection, bm25_retrieve)
        rerank_retrieve = make_rerank_retriever(hybrid_retrieve)
        for name, fn in {"Hybrid": hybrid_retrieve, "Rerank": rerank_retrieve}.items():
            h1 = evaluate(collection, top_k=1, verbose=False, dataset=dataset, retrieve_fn=fn)
            h3 = evaluate(collection, top_k=3, verbose=False, dataset=dataset, retrieve_fn=fn)
            print(f"{chunk_name:8} chunks | {name:6} | Hit@1 = {h1:.0%} | Hit@3 = {h3:.0%}")

build_index("clinic_info.pdf")