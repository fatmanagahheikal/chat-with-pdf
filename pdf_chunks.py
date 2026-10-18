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

def stitch_pages(parts):
    full_text = ""
    ruler = []
    for page, text in parts:
        if full_text:
            full_text += " "
        start = len(full_text)
        full_text += text
        ruler.append((start, len(full_text), page))
    return full_text, ruler

def page_at(ruler, position):
    for start, end, page in ruler:
        if position < end:
            return page
    return ruler[-1][2]

def load_pdf_chunks(pdf_path, chunk_size=500, chunk_overlap=150):
    reader = PdfReader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ". ", " ", ""],
        keep_separator="end",
        add_start_index=True,
    )
    page_texts = [page.extract_text() for page in reader.pages]
    repeated = find_repeated_lines(page_texts)

    parts_in_order = []
    section = "body"
    for i, text in enumerate(page_texts):
        parts, section = split_page_sections(remove_repeated_lines(text, repeated), section)
        for part_section, part_text in parts:
            cleaned = clean_text(part_text).strip()
            if cleaned:
                parts_in_order.append((part_section, i + 1, cleaned))

    runs = []
    for part_section, page, text in parts_in_order:
        if runs and runs[-1]["section"] == part_section:
            runs[-1]["parts"].append((page, text))
        else:
            runs.append({"section": part_section, "parts": [(page, text)]})

    all_chunks = []
    for run in runs:
        full_text, ruler = stitch_pages(run["parts"])
        for doc in splitter.create_documents([full_text]):
            start = max(doc.metadata["start_index"], 0)
            end = start + len(doc.page_content) - 1
            all_chunks.append({
                "text": doc.page_content,
                "page": page_at(ruler, start),
                "page_end": page_at(ruler, end),
                "section": run["section"],
            })
    return all_chunks

if __name__ == "__main__":
    for pdf in ["paper.pdf", "clinic_info.pdf"]:
        chunks = load_pdf_chunks(pdf)
        crossing = [c for c in chunks if c["page"] != c["page_end"]]
        print(f"{pdf}: {len(chunks)} chunks, {len(crossing)} cross a page boundary")

    print("\n--- Pig skin chunk ---")
    for c in load_pdf_chunks("paper.pdf"):
        if "scattering properties" in c["text"]:
            print(f"Pages {c['page']} to {c['page_end']}:")
            print(c["text"])
            print()