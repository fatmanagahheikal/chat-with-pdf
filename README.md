# Chat with your PDF

A RAG application that answers questions about any PDF and cites the page it used.
Every step was measured with a golden dataset before it was kept.

## How it works

1. Read the PDF with pypdf, remove repeated header and footer lines automatically,
   and split the text into sections (body, references, author bios).
2. Split the text into chunks of 500 characters with 150 characters overlap.
3. Embed the chunks with BAAI/bge-small-en-v1.5 and store them in ChromaDB
   with page and section metadata.
4. For each question, search the body section with two methods:
   semantic search (embeddings) and BM25 (keywords).
5. Merge the two ranked lists with Reciprocal Rank Fusion (RRF, k=60)
   and keep the top 20 candidates.
6. Re-rank the 20 candidates with a cross-encoder (ms-marco-MiniLM-L-6-v2)
   and keep the top 3.
7. Send the top 3 chunks to an LLM (deepseek-v4.1-flash via OpenRouter).
   It answers only from these chunks and cites the page, or says "I don't know".

## Results

Evaluated on two documents: my IEEE Access paper (40 questions)
and a dental clinic handbook (20 questions).
Hit@k means the answer text is inside one of the top k retrieved chunks.

| Method | Paper Hit@1 | Paper Hit@3 | Clinic Hit@1 | Clinic Hit@3 | Time per question |
|---|---|---|---|---|---|
| Semantic, all-MiniLM-L6-v2 (baseline) | 32% | 52% | 95% | 100% | - |
| Semantic, bge-small-en-v1.5 | 45% | 72% | 100% | 100% | 157 ms |
| BM25 | 50% | 72% | 60% | 90% | 3 ms |
| Hybrid (RRF) | 60% | 82% | 95% | 95% | 163 ms |
| **Hybrid + Re-ranking** | **70%** | **90%** | **100%** | **100%** | 3.3 s |

Time is measured on a laptop CPU without a GPU.
Full details, including header cleaning and chunking experiments, are in [RESULTS.md](RESULTS.md).

## Key findings

1. A stronger embedding model gave the biggest single gain (paper Hit@3 from 52% to 72%).
2. BM25 wins on the paper (rare technical terms) but loses on the clinic
   (questions use different words than the document). Hybrid + re-ranking is the best on both.
3. Re-ranking costs about 3 seconds per question on CPU, in exchange for 8 more points of Hit@3.
4. Semantic chunking and LLM chunking did not beat simple fixed chunking once re-ranking was used.
   LLM chunking was also not reproducible (18 chunks in one run, 23 in another).

## Known limitations

1. Tables and figures in the paper are images, so their content is not extracted.
2. pypdf sometimes drops the space between two words.
3. The evaluation set is small (60 questions), so a difference of one question is not meaningful.

## Run locally

```bash
git clone https://github.com/fatmanagahheikal/chat-with-pdf.git
cd chat-with-pdf
conda create -n genai python=3.11
conda activate genai
pip install -r requirements.txt
```

Create a file named `.env` in the project folder with your OpenRouter key:

```
OPENROUTER_API_KEY=your_key_here
```

Then run:

```bash
python app.py
```

## Main files

| File | What it does |
|---|---|
| `app.py` | Gradio interface |
| `pdf_chunks.py` | Reads the PDF, cleans it, detects sections, and splits it into chunks |
| `build_index_pdf.py` | Embeds the chunks and stores them in ChromaDB |
| `bm25_search.py` | Keyword search |
| `hybrid_search.py` | Merges semantic and BM25 results with RRF |
| `rerank_search.py` | Re-ranks the candidates with a cross-encoder |
| `eval_compare.py` | Compares all search methods on both documents |
| `golden_paper.py`, `golden_dataset.py` | The golden questions and answers |

`paper.pdf` is my paper published in IEEE Access under the CC BY 4.0 license.

## Author

Fatma N. Heikal · [LinkedIn](https://www.linkedin.com/in/fatma-heikal-029b90203)



## Recent Update

- Chunks can now cross page boundaries (pages are stitched before splitting), and sources show page ranges like "page 4-5". Details in `RESULTS.md`.
**App check:** "Why did you use pig skin?" now cites "page 4-5" and gets the full optical sentence, but still misses the anatomical reason (previous chunk, not in top 3). Needs answer-level (G) evaluation later.