# Retrieval Results

Setup: chunk_size=500, chunk_overlap=150, search in body section only.
Hardware: laptop CPU (no GPU).

## paper.pdf (40 questions, 161 body chunks)

| Method | Hit@1 | Hit@3 | Time per question |
|---|---|---|---|
| Semantic (all-MiniLM-L6-v2) | 32% | 52% | - |
| Semantic (bge-small-en-v1.5) | 45% | 72% | 157 ms |
| BM25 | 50% | 72% | 3 ms |
| Hybrid (RRF, k=60, 20 candidates) | 60% | 82% | 163 ms |
| Hybrid + Re-ranking (ms-marco-MiniLM-L-6-v2) | 70% | 90% | 3328 ms |

## clinic_info.pdf (20 questions, 28 chunks)

| Method | Hit@1 | Hit@3 |
|---|---|---|
| Semantic (bge-small-en-v1.5) | 100% | 100% |
| BM25 | 60% | 90% |
| Hybrid (RRF) | 95% | 95% |
| Hybrid + Re-ranking | 100% | 100% |

## Decision

Hybrid + Re-ranking is the best on both files.
Cost: about 3.3 seconds per question on CPU (20x slower than Hybrid).

## Header/footer cleaning

Lines repeated on 50% or more of the pages (digits ignored) are removed automatically.
Removed in paper.pdf: running title line and "VOLUME 14, 2026 + page number".
Removed in clinic_info.pdf: "Smile Dental Clinic - Patient Handbook 2026 Page X".

| File | Re-ranking Hit@1 (before / after) | Re-ranking Hit@3 (before / after) |
|---|---|---|
| paper.pdf | 70% / 72% | 90% / 90% |
| clinic_info.pdf | 100% / 95% | 100% / 100% |

Decision: kept. The effect on retrieval is about one question in each direction,
but chunks sent to the LLM are cleaner, and it works on any PDF automatically.