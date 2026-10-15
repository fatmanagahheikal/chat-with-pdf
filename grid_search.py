from build_index_pdf import build_index
from evaluate_retrieval import evaluate

PDF_PATH = "clinic_info.pdf"

configs = [
    (100, 0),
    (200, 0),
    (200, 50),
    (300, 50),
    (500, 150),
    (800, 200),
    (1200, 200),
]

results = []
for chunk_size, chunk_overlap in configs:
    collection = build_index(PDF_PATH, chunk_size, chunk_overlap)
    hit_at_1 = evaluate(collection, top_k=1, verbose=False)
    hit_at_3 = evaluate(collection, top_k=3, verbose=False)
    results.append((chunk_size, chunk_overlap, collection.count(), hit_at_1, hit_at_3))

print("\n===== GRID SEARCH RESULTS =====")
print("size | overlap | chunks | Hit@1 | Hit@3")
for size, overlap, count, h1, h3 in results:
    print(f"{size:4} | {overlap:7} | {count:6} | {h1:5.0%} | {h3:5.0%}")

build_index(PDF_PATH)
print("\nRebuilt the default index (500 / 150) for the app.")
