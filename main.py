import os
import io
import requests
import fitz  # PyMuPDF
import docx
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import google.generativeai as genai

# --- Gemini API Setup ---
GEMINI_API_KEY = "AIzaSyANhg2f7oPjkuMAttXl_r_irN3u0_mTscI"
genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_embedding(text: str) -> List[float]:
    model = genai.get_model("embedding-001")
    resp = model.embed_content([text])
    return resp["embedding"]

def gemini_answer(question: str, context: str) -> str:
    model = genai.GenerativeModel("gemini-pro")
    prompt = (
        f"Given the following context from a policy/contract:\n\n{context}\n\n"
        f"Answer the question: '{question}'\n"
        "If possible, cite the relevant clause(s) and explain your reasoning."
    )
    resp = model.generate_content(prompt)
    return resp.text

# --- Document Parsing ---
def extract_text_from_pdf(url: str) -> str:
    response = requests.get(url)
    doc = fitz.open(stream=response.content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(url: str) -> str:
    response = requests.get(url)
    doc = docx.Document(io.BytesIO(response.content))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_email(url: str) -> str:
    response = requests.get(url)
    from email import message_from_bytes
    msg = message_from_bytes(response.content)
    return msg.get_payload(decode=True).decode(errors="ignore")

def extract_text(url: str) -> str:
    if url.endswith(".pdf"):
        return extract_text_from_pdf(url)
    elif url.endswith(".docx"):
        return extract_text_from_docx(url)
    elif url.endswith(".eml"):
        return extract_text_from_email(url)
    else:
        raise ValueError("Unsupported file type")

# --- Chunking ---
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# --- FAISS Vector Store ---
class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []

    def add(self, embeddings: List[List[float]], chunks: List[str]):
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunks.extend(chunks)

    def search(self, embedding: List[float], top_k: int = 3):
        D, I = self.index.search(np.array([embedding]).astype("float32"), top_k)
        return [self.chunks[i] for i in I[0]]

# --- FastAPI Setup ---
app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
def run_query(req: QueryRequest):
    # 1. Download and parse document
    try:
        text = extract_text(req.documents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Document parsing failed: {e}")

    # 2. Chunk and embed
    chunks = chunk_text(text)
    embeddings = [get_gemini_embedding(chunk) for chunk in chunks]
    dim = len(embeddings[0])
    store = VectorStore(dim)
    store.add(embeddings, chunks)

    # 3. For each question: retrieve, reason, answer
    answers = []
    for q in req.questions:
        q_emb = get_gemini_embedding(q)
        relevant_chunks = store.search(q_emb, top_k=3)
        context = "\n---\n".join(relevant_chunks)
        answer = gemini_answer(q, context)
        answers.append(answer.strip())

    return {"answers": answers}

# --- For Render: use $PORT env var if present ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)