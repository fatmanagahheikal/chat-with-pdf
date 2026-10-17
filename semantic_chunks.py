import re
import numpy as np
from pypdf import PdfReader
from pdf_chunks import clean_text, find_repeated_lines, remove_repeated_lines, split_page_sections
from build_index_pdf import embedder

def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]

def load_semantic_chunks(pdf_path, percentile=25, max_chars=1000, min_chars=200):
    page_texts = [p.extract_text() for p in PdfReader(pdf_path).pages]
    repeated = find_repeated_lines(page_texts)
    sentences = []
    section = "body"
    for i, text in enumerate(page_texts):
        parts, section = split_page_sections(remove_repeated_lines(text, repeated), section)
        for part_section, part_text in parts:
            for s in split_sentences(clean_text(part_text)):
                sentences.append({"text": s, "page": i + 1, "section": part_section})

    vectors = embedder.encode([s["text"] for s in sentences], normalize_embeddings=True)
    sims = [float(vectors[i] @ vectors[i + 1]) for i in range(len(sentences) - 1)]
    threshold = np.percentile(sims, percentile)

    chunks = []
    current = sentences[0].copy()
    for i in range(1, len(sentences)):
        s = sentences[i]
        topic_change = sims[i - 1] < threshold and len(current["text"]) >= min_chars
        too_long = len(current["text"]) + len(s["text"]) > max_chars
        new_section = s["section"] != current["section"]
        if topic_change or too_long or new_section:
            chunks.append(current)
            current = s.copy()
        else:
            current["text"] += " " + s["text"]
    chunks.append(current)
    return chunks

if __name__ == "__main__":
    for path in ["paper.pdf", "clinic_info.pdf"]:
        chunks = load_semantic_chunks(path)
        sizes = [len(c["text"]) for c in chunks]
        print(f"\n{path}: {len(chunks)} chunks, average {sum(sizes) // len(sizes)} chars")
        if path == "clinic_info.pdf":
            for c in chunks[:5]:
                print(f"\n[Page {c['page']}] {c['text'][:150]}...")