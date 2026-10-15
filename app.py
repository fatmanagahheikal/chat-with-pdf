import os
import chromadb
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("clinic_pdf")
print(f"Loaded {collection.count()} chunks from the database")

def retrieve(question, top_k=3):
    q = embedder.encode(question, normalize_embeddings=True)
    results = collection.query(query_embeddings=[q.tolist()], n_results=top_k)
    docs = results["documents"][0]
    distances = results["distances"][0]
    metas = results["metadatas"][0]
    return [(d, 1 - dist / 2, m["page"]) for d, dist, m in zip(docs, distances, metas)]

def answer_question(question):
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

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Your question"),
    outputs=gr.Textbox(label="Answer"),
    title="Smile Dental Assistant 🦷",
)

demo.launch()