import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

REFERENCES_HEADINGS = {"REFERENCES", "BIBLIOGRAPHY"}
BIO_START = re.compile(r"^[A-Z][A-Z.\- ]{3,}\s+(received|\((Senior )?Member, IEEE\))")

def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("Dr. ", "Dr ")
    return text

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
    all_chunks = []
    section = "body"
    for i, page in enumerate(reader.pages):
        parts, section = split_page_sections(page.extract_text(), section)
        for part_section, part_text in parts:
            for chunk in splitter.split_text(clean_text(part_text)):
                all_chunks.append({"text": chunk, "page": i + 1, "section": part_section})
    return all_chunks

if __name__ == "__main__":
    chunks = load_pdf_chunks("clinic_info.pdf")
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"[Page {c['page']}] [{c['section']}] {c['text'][:70]}...")