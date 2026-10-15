from pdf_chunks import load_pdf_chunks
from golden_paper import GOLDEN_PAPER

chunks = load_pdf_chunks("paper.pdf")
found_count = 0

for item in GOLDEN_PAPER:
    found_pages = [c["page"] for c in chunks if item["answer"].lower() in c["text"].lower()]
    if found_pages:
        found_count += 1
    else:
        print(f"NOT FOUND ANYWHERE | answer: {item['answer']} | {item['question']}")

print(f"\n{found_count}/{len(GOLDEN_PAPER)} answers exist somewhere in the chunks")