from collections import Counter
from pdf_chunks import load_pdf_chunks

chunks = load_pdf_chunks("paper.pdf")
print(Counter(c["section"] for c in chunks))

body_chunks = [c for c in chunks if c["section"] == "body"]
print(f"\nLast 'body' chunk (page {body_chunks[-1]['page']}):\n{body_chunks[-1]['text'][-150:]}")

for name in ["references", "bio"]:
    first = next((c for c in chunks if c["section"] == name), None)
    if first:
        print(f"\nFirst '{name}' chunk (page {first['page']}):\n{first['text'][:150]}")
    else:
        print(f"\nNo '{name}' chunks found!")