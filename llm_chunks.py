import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from pdf_chunks import clean_text, find_repeated_lines, remove_repeated_lines, split_page_sections
from semantic_chunks import split_sentences

load_dotenv()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
MODEL = "deepseek/deepseek-v4.1-flash"

PROMPT = """Below are numbered sentences from one page of a document.
Group them into chunks so that each chunk covers one topic.
Return ONLY a JSON list with the number of the sentence where each chunk starts.
The first number must be 0. Example: [0, 4, 9]

{sentences}"""

def ask_llm_for_starts(sentences):
    numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(sentences=numbered)}],
        temperature=0,
        max_tokens=1000,
    )
    text = response.choices[0].message.content or ""
    match = re.search(r"\[[\d,\s]*\]", text)
    if not match:
        return [0]
    return sorted({n for n in json.loads(match.group()) if 0 <= n < len(sentences)} | {0})

def group_sentences(sentences, starts, max_chars):
    chunks = []
    for a, b in zip(starts, starts[1:] + [len(sentences)]):
        current = ""
        for s in sentences[a:b]:
            if current and len(current) + len(s) > max_chars:
                chunks.append(current)
                current = s
            else:
                current = (current + " " + s).strip()
        if current:
            chunks.append(current)
    return chunks

def load_llm_chunks(pdf_path, max_chars=1000):
    page_texts = [p.extract_text() for p in PdfReader(pdf_path).pages]
    repeated = find_repeated_lines(page_texts)
    all_chunks = []
    section = "body"
    for i, text in enumerate(page_texts):
        parts, section = split_page_sections(remove_repeated_lines(text, repeated), section)
        for part_section, part_text in parts:
            sentences = split_sentences(clean_text(part_text))
            if not sentences:
                continue
            starts = ask_llm_for_starts(sentences) if part_section == "body" else [0]
            for chunk in group_sentences(sentences, starts, max_chars):
                all_chunks.append({"text": chunk, "page": i + 1, "section": part_section})
        print(f"Page {i + 1} done")
    return all_chunks

if __name__ == "__main__":
    chunks = load_llm_chunks("clinic_info.pdf")
    sizes = [len(c["text"]) for c in chunks]
    print(f"\nclinic_info.pdf: {len(chunks)} chunks, average {sum(sizes) // len(sizes)} chars")
    for c in chunks[:5]:
        print(f"\n[Page {c['page']}] {c['text'][:150]}...")