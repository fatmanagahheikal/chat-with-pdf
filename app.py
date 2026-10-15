import os
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from build_index_pdf import build_index, embedder, db

load_dotenv()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

def process_pdf(pdf_path):
    if pdf_path is None:
        return "Please upload a PDF first."
    collection = build_index(pdf_path)
    return f"✅ Ready! Indexed {collection.count()} chunks. Ask your question."

def retrieve(question, top_k=3):
    collection = db.get_collection("clinic_pdf")
    q = embedder.encode(question, normalize_embeddings=True)
    results = collection.query(query_embeddings=[q.tolist()], n_results=top_k)
    docs = results["documents"][0]
    distances = results["distances"][0]
    metas = results["metadatas"][0]
    return [(d, 1 - dist / 2, m["page"]) for d, dist, m in zip(docs, distances, metas)]

def answer_question(question):
    if not question.strip():
        return "Please type a question first."
    results = retrieve(question)
    context = "\n\n".join(f"[Page {page}]\n{chunk}" for chunk, score, page in results)
    prompt = f"""Answer the question using ONLY the context below.

- If the context contains the answer: answer clearly, then end with the page number(s) you used, like: (Source: page 4)
- If the context does not contain the answer: reply with only "I don't know", with no source.

Context:
{context}

Question: {question}"""
    response = client.chat.completions.create(
        model="deepseek/deepseek-v4.1-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=1000,
    )
    return response.choices[0].message.content

custom_css = """
.header {
    background: linear-gradient(135deg, #0f172a 0%, #1e40af 55%, #0ea5e9 100%);
    padding: 32px 24px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(30, 64, 175, 0.35);
}
.header h1 {
    color: #ffffff;
    font-size: 2.3em;
    font-weight: 800;
    margin: 0;
}
.header p {
    color: #bfdbfe;
    font-size: 1.05em;
    margin: 10px 0 0 0;
}
.badges span {
    display: inline-block;
    background: rgba(255, 255, 255, 0.15);
    color: #e0f2fe;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.85em;
    margin: 12px 4px 0 4px;
}
.footer {
    text-align: center;
    opacity: 0.7;
    font-size: 0.9em;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks() as demo:
    gr.HTML("""
    <div class="header">
        <h1>📄 Chat with your PDF</h1>
        <p>Upload any PDF, ask questions, and get answers with page sources.</p>
        <div class="badges">
            <span>RAG</span><span>ChromaDB</span><span>Sentence Transformers</span><span>Gradio</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ Upload your document")
            pdf_input = gr.File(label="Upload one PDF", file_types=[".pdf"], type="filepath")
            build_btn = gr.Button("⚙️ Process PDF", variant="primary", size="lg")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 2️⃣ Ask a question")
            question_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. Which doctor treats children?",
                lines=2,
            )
            ask_btn = gr.Button("🔍 Ask", variant="primary", size="lg")
            answer_output = gr.Textbox(label="Answer", lines=6, interactive=False)
            gr.Examples(
                examples=[
                    "Which doctor treats children?",
                    "How long does a dental implant take?",
                    "What is the emergency fee outside working hours?",
                ],
                inputs=question_input,
            )

    gr.HTML('<div class="footer">Built by Fatma Heikal · Answers come only from your document</div>')

    build_btn.click(fn=process_pdf, inputs=pdf_input, outputs=status)
    ask_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)
    question_input.submit(fn=answer_question, inputs=question_input, outputs=answer_output)

demo.launch(theme=theme, css=custom_css)