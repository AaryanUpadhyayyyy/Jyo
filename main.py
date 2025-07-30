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
    # Try to determine file type from URL or content
    if url.endswith(".pdf") or "pdf" in url.lower():
        return extract_text_from_pdf(url)
    elif url.endswith(".docx") or "docx" in url.lower():
        return extract_text_from_docx(url)
    elif url.endswith(".eml") or "eml" in url.lower():
        return extract_text_from_email(url)
    else:
        # Default to PDF for URLs without clear extension
        try:
            return extract_text_from_pdf(url)
        except Exception as e:
            raise ValueError(f"Could not parse document. Tried PDF format but failed: {e}")

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

@app.get("/")
def read_root():
    return {"message": "LLM-Powered Query Retrieval System is running!", "endpoint": "/api/v1/hackrx/run"}

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
def run_query(req: QueryRequest):
    # 1. Download and parse document
    try:
        print(f"Attempting to parse document from: {req.documents}")
        text = extract_text(req.documents)
        print(f"Successfully extracted {len(text)} characters from document")
    except Exception as e:
        print(f"Document parsing error: {e}")
        raise HTTPException(status_code=400, detail=f"Document parsing failed: {e}")

    # 2. Chunk and embed
    try:
        chunks = chunk_text(text)
        print(f"Created {len(chunks)} chunks from document")
        
        embeddings = [get_gemini_embedding(chunk) for chunk in chunks]
        print(f"Generated {len(embeddings)} embeddings")
        
        dim = len(embeddings[0])
        store = VectorStore(dim)
        store.add(embeddings, chunks)
        print("Successfully created vector store")
    except Exception as e:
        print(f"Embedding/vector store error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    # 3. For each question: retrieve, reason, answer
    answers = []
    for i, q in enumerate(req.questions):
        try:
            print(f"Processing question {i+1}/{len(req.questions)}: {q[:50]}...")
            q_emb = get_gemini_embedding(q)
            relevant_chunks = store.search(q_emb, top_k=3)
            context = "\n---\n".join(relevant_chunks)
            answer = gemini_answer(q, context)
            answers.append(answer.strip())
            print(f"Successfully answered question {i+1}")
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            answers.append(f"Error processing question: {e}")

    return {"answers": answers}

# --- For Render: use $PORT env var if present ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
