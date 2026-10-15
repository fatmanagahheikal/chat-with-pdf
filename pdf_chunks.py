from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("Dr. ", "Dr ")
    return text

def load_pdf_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        separators=["\n\n", ". ", " ", ""],
        keep_separator="end",
    )
    all_chunks = []
    for i, page in enumerate(reader.pages):
        page_text = clean_text(page.extract_text())
        for chunk in splitter.split_text(page_text):
            all_chunks.append({"text": chunk, "page": i + 1})
    return all_chunks

if __name__ == "__main__":
    chunks = load_pdf_chunks("clinic_info.pdf")
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"[Page {c['page']}] {c['text'][:70]}...")