import re
from collections import Counter
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

REFERENCES_HEADINGS = {"REFERENCES", "BIBLIOGRAPHY"}
BIO_START = re.compile(r"^[A-Z][A-Z.\- ]{3,}\s+(received|\((Senior )?Member, IEEE\))")

def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("Dr. ", "Dr ")
    return text

def normalize_line(line):
    line = re.sub(r"\d+", "", line)
    return " ".join(line.split())

def find_repeated_lines(page_texts, min_ratio=0.5):
    counts = Counter()
    for text in page_texts:
        lines = {normalize_line(l) for l in text.split("\n")}
        lines.discard("")
        counts.update(lines)
    min_pages = max(3, int(len(page_texts) * min_ratio))
    return {line for line, n in counts.items() if n >= min_pages}

def remove_repeated_lines(text, repeated):
    kept = [l for l in text.split("\n") if normalize_line(l) not in repeated]
    return "\n".join(kept)

def split_page_sections(page_text, section):
    parts = []
    current_lines = []
    for line in page_text.split("\n"):
        stripped = line.strip()
        new_section = section
        if section == "body" and stripped.upper() in REFERENCES_HEADINGS:
            new_section = "references"
        elif section == "references" and BIO_START.match(stripped):
            new_section = "bio"
        if new_section != section:
            parts.append((section, "\n".join(current_lines)))
            current_lines = []
            section = new_section
        current_lines.append(line)
    parts.append((section, "\n".join(current_lines)))
    return parts, section

def load_pdf_chunks(pdf_path, chunk_size=500, chunk_overlap=150):
    reader = PdfReader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ". ", " ", ""],
        keep_separator="end",
    )
    page_texts = [page.extract_text() for page in reader.pages]
    repeated = find_repeated_lines(page_texts)
    all_chunks = []
    section = "body"
    for i, text in enumerate(page_texts):
        parts, section = split_page_sections(remove_repeated_lines(text, repeated), section)
        for part_section, part_text in parts:
            for chunk in splitter.split_text(clean_text(part_text)):
                all_chunks.append({"text": chunk, "page": i + 1, "section": part_section})
    return all_chunks

if __name__ == "__main__":
    for pdf_path in ["paper.pdf", "clinic_info.pdf"]:
        page_texts = [p.extract_text() for p in PdfReader(pdf_path).pages]
        print(f"\n===== {pdf_path} =====")
        for line in find_repeated_lines(page_texts):
            print("REMOVED:", line)
        print(f"Total chunks: {len(load_pdf_chunks(pdf_path))}")