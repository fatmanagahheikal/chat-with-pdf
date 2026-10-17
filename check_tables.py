from pdf_chunks import load_pdf_chunks

CHECKS = [
    ("Table 1", "Key limitation"),
    ("Table 1", "Perfusion mapping"),
    ("Table 2", "RobustScaler"),
    ("Table 2", "Out-of-fold"),
    ("Figure 3", "12,941,396"),
    ("Figure 4", "reduction=8"),
]

chunks = load_pdf_chunks("paper.pdf")
for where, text in CHECKS:
    found = [c["page"] for c in chunks if text.lower() in c["text"].lower()]
    status = "FOUND" if found else "LOST "
    print(f"{status} | {where:8} | {text} | pages {found}")