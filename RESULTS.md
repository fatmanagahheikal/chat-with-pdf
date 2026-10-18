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

## Known limitation: tables and figures

Tables and figures in paper.pdf are embedded as images, so pypdf reads only their captions.
Checked with check_tables.py: 6 of 6 facts from tables and figures are missing from the chunks
(for example RobustScaler in Table 2, and 12,941,396 parameters in Figure 3).
The golden dataset has no questions from tables, so the Hit Rate does not show this gap.
Possible fixes later: OCR, a vision LLM that describes each table and figure, or GROBID.
Before fixing it, add golden questions whose answers are only in tables, to measure the fix.


## Semantic chunking

Split text into sentences, embed each sentence, and cut where the similarity between
neighbouring sentences is in the lowest 25%. Minimum 200 chars, maximum 1000 chars.
paper.pdf: 151 chunks (average 507 chars). clinic_info.pdf: 16 chunks (average 553 chars).

| File | Chunking | Hybrid Hit@1 | Hybrid Hit@3 | Rerank Hit@1 | Rerank Hit@3 |
|---|---|---|---|---|---|
| paper.pdf | Fixed (500/150) | 62% | 82% | 72% | 90% |
| paper.pdf | Semantic | 50% | 82% | 72% | 92% |
| clinic_info.pdf | Fixed (500/150) | 90% | 100% | 95% | 100% |
| clinic_info.pdf | Semantic | 85% | 100% | 95% | 100% |

Decision: keep fixed chunking. With re-ranking the results are the same (one question difference),
Hybrid alone gets worse, and semantic chunking is slower to build (it embeds every sentence).

Known issue: pypdf sometimes drops the space between two words (for example "glovesand").

## LLM chunking

The LLM (deepseek-v4.1-flash) gets the numbered sentences of each page and returns
the numbers where a new topic starts. The text itself is never rewritten.

| File | Chunking | Chunks | Hybrid Hit@1 | Hybrid Hit@3 | Rerank Hit@1 | Rerank Hit@3 |
|---|---|---|---|---|---|---|
| paper.pdf | Fixed (500/150) | 219 | 62% | 82% | 72% | 90% |
| paper.pdf | Semantic | 151 | 50% | 82% | 72% | 92% |
| paper.pdf | LLM | 93 | 60% | 80% | 72% | 90% |
| clinic_info.pdf | Fixed (500/150) | 27 | 90% | 100% | 95% | 100% |
| clinic_info.pdf | Semantic | 16 | 85% | 100% | 95% | 100% |
| clinic_info.pdf | LLM | 23 | 95% | 100% | 100% | 100% |

LLM chunking is not reproducible: the same clinic file gave 18 chunks in one run and 23 in another.

Decision: keep fixed chunking. With re-ranking, all three methods are within one question.
Fixed chunking is free, fast and gives the same chunks every time.
LLM chunking kept headings with their content and was one question better on the clinic,
so it is worth testing again on more documents and more questions.