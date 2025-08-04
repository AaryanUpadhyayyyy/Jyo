import os
import io
import requests
import fitz  # PyMuPDF
import docx
import faiss
import numpy as np
import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Gemini API Setup ---
# A single key is used for all Gemini services.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set.")
    raise ValueError("GEMINI_API_KEY environment variable not set. Cannot proceed.")

# Configure the API client once
genai.configure(api_key=GEMINI_API_KEY)
logger.info(f"DEBUG: Using Gemini API Key starting with: {GEMINI_API_KEY[:5]}*****")

# --- Feature Flags ---
ENABLE_LLM_RERANKING = os.getenv("ENABLE_LLM_RERANKING", "true").lower() == "true"
logger.info(f"Feature Flag: ENABLE_LLM_RERANKING is set to {ENABLE_LLM_RERANKING}")

def get_gemini_embedding(text: str, retries: int = 3, backoff_factor: float = 0.5) -> List[float]:
    """Generates Gemini embeddings with exponential backoff."""
    for i in range(retries):
        try:
            resp = genai.embed_content(model="models/embedding-001", content=[text])
            embedding_vector = resp["embedding"]
            if isinstance(embedding_vector, list) and len(embedding_vector) == 1 and isinstance(embedding_vector[0], list):
                embedding_vector = embedding_vector[0]
            return embedding_vector
        except Exception as e:
            logger.warning(f"Embedding generation failed (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                sleep_time = backoff_factor * (2 ** i)
                logger.info(f"Retrying after {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Final embedding generation failed after {retries} retries: {e}")
                raise

def re_rank_chunks_with_llm(query: str, chunks: List[str], top_n_rerank: int = 5, retries: int = 3, backoff_factor: float = 0.5) -> List[str]:
    """Uses Gemini to re-rank a list of chunks based on their relevance to the query."""
    if not chunks:
        return []

    model_name = "gemini-2.5-flash-preview-05-20"
    rerank_prompt = (
        f"Given the user query and a list of text segments, identify the {top_n_rerank} most relevant segments.\n"
        f"Return ONLY the content of the selected segments, each on a new line, exactly as they appear in the input list.\n"
        f"If fewer than {top_n_rerank} relevant segments are found, return all relevant ones.\n\n"
        f"Query: '{query}'\n\n"
        f"Text Segments (each prefixed with 'Segment X:'):"
    )
    for i, chunk in enumerate(chunks):
        rerank_prompt += f"\nSegment {i+1}: {chunk}"
    
    for i in range(retries):
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(rerank_prompt)
            ranked_segments_text = resp.text.strip()
            re_ranked_chunks = [line.strip() for line in ranked_segments_text.split('\n') if line.strip()]
            final_re_ranked = [original for original in chunks if original.strip() in re_ranked_chunks][:top_n_rerank]
            if len(final_re_ranked) < top_n_rerank and len(chunks) > 0:
                logger.warning(f"Re-ranking returned fewer than {top_n_rerank} chunks. Falling back to original top N.")
            return final_re_ranked if final_re_ranked else chunks[:top_n_rerank]
        except Exception as e:
            logger.warning(f"Re-ranking failed (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                time.sleep(backoff_factor * (2 ** i))
            else:
                logger.error("Final re-ranking failed after all retries. Returning original chunks.")
                return chunks[:top_n_rerank]

def summarize_context(context: str, retries: int = 3, backoff_factor: float = 0.5) -> str:
    """Uses Gemini to provide a concise, 3-4 line summary of the context."""
    for i in range(retries):
        try:
            model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
            summary_prompt = (
                "Given the following text, provide a concise summary of the key points in 3-4 lines.\n\n"
                f"**Text:**\n{context}"
            )
            resp = model.generate_content(summary_prompt)
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"Summarization failed (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                time.sleep(backoff_factor * (2 ** i))
            else:
                logger.error("Summarization failed after retries. Returning error message.")
                return "Failed to summarize context."

def gemini_answer(question: str, context: str, retries: int = 3, backoff_factor: float = 0.5) -> str:
    """Uses Gemini to answer a question based on provided context."""
    for i in range(retries):
        try:
            model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
            prompt = (
                f"Given the following context from a policy/contract:\n\n{context}\n\n"
                f"Answer the question: '{question}' concisely and directly in a complete sentence. "
                f"If the answer is not in the context, say 'The provided text does not contain this information.'."
                f"Do not add any additional commentary, reasoning, or quotes unless they are the direct answer."
            )
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"Answer failed (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                time.sleep(backoff_factor * (2 ** i))
            else:
                logger.error("Answer generation failed after retries.")
                return "Error processing question: Failed to get an answer after multiple attempts due to API issues."

# --- Document Parsing Functions ---
def extract_text_from_pdf(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    doc = fitz.open(stream=response.content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_text_from_docx(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    doc = docx.Document(io.BytesIO(response.content))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_email(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    from email import message_from_bytes
    msg = message_from_bytes(response.content)
    payload = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                payload = part.get_payload(decode=True).decode(errors="ignore")
                break
    else:
        payload = msg.get_payload(decode=True).decode(errors="ignore")
    return payload

def extract_text(url: str) -> str:
    logger.info(f"Extracting text from: {url}")
    try:
        if url.lower().endswith(".pdf"):
            return extract_text_from_pdf(url)
        elif url.lower().endswith(".docx"):
            return extract_text_from_docx(url)
        elif url.lower().endswith(".eml"):
            return extract_text_from_email(url)
        else:
            logger.warning(f"File extension unclear for {url}, assuming PDF.")
            return extract_text_from_pdf(url)
    except requests.exceptions.RequestException as req_e:
        logger.error(f"Network error fetching document: {req_e}")
        raise ValueError(f"Failed to fetch document from URL: {req_e}")
    except Exception as e:
        logger.error(f"Failed to extract text: {e}")
        raise ValueError(f"Could not parse document: {e}")

def chunk_text(text: str, max_chunk_words: int = 700, chunk_overlap_words: int = 100) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks, current_chunk_words = [], []
    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        if len(current_chunk_words) + len(paragraph_words) > max_chunk_words and current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
            overlap_words = current_chunk_words[max(0, len(current_chunk_words)-chunk_overlap_words):]
            current_chunk_words = overlap_words
        if len(paragraph_words) > max_chunk_words:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []
            for i in range(0, len(paragraph_words), max_chunk_words-chunk_overlap_words):
                sub_chunk = paragraph_words[i:i+max_chunk_words]
                chunks.append(" ".join(sub_chunk))
            current_chunk_words = paragraph_words[len(paragraph_words)-chunk_overlap_words:] if len(paragraph_words) >= chunk_overlap_words else []
        else:
            current_chunk_words += paragraph_words
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
    return chunks

def keyword_search(query: str, chunks: List[str], top_n_keywords: int = 2) -> List[str]:
    query_words = [w.lower() for w in query.split() if len(w) > 2]
    found = []
    for chunk in chunks:
        if any(word in chunk.lower() for word in query_words):
            found.append(chunk)
            if len(found) >= top_n_keywords:
                break
    return found

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []
    def add(self, embeddings: List[List[float]], chunks: List[str]):
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunks.extend(chunks)
    def search(self, embedding: List[float], top_k: int = 5):
        D, I = self.index.search(np.array([embedding]).astype("float32"), top_k)
        return [self.chunks[i] for i in I[0]]

app = FastAPI(title="LLM-Powered Document Q&A System", version="1.0.0")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]
class AnswerWithContext(BaseModel):
    answer: str
    context: str
class QueryResponse(BaseModel):
    answers: List[AnswerWithContext]

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
def run_query(req: QueryRequest):
    logger.info(f"Received request for document: {req.documents} with {len(req.questions)} questions.")
    try:
        text = extract_text(req.documents)
        if not text:
            raise HTTPException(status_code=400, detail="Extracted text is empty.")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Document parsing failed: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    try:
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from document.")
        embeddings = [get_gemini_embedding(chunk) for chunk in chunks]
        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings.")
        dim = len(embeddings[0])
        if not all(len(emb) == dim for emb in embeddings):
            raise HTTPException(status_code=500, detail="Inconsistent embedding dimensions.")
        store = VectorStore(dim)
        store.add(embeddings, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    answers_with_context = []
    for i, q in enumerate(req.questions):
        try:
            q_emb = get_gemini_embedding(q)
            faiss_relevant_chunks = store.search(q_emb)
            keyword_relevant_chunks = keyword_search(q, chunks)
            combined = list(dict.fromkeys(faiss_relevant_chunks + keyword_relevant_chunks))
            
            final_context = "\n---\n".join(combined[:5])
            
            answer = gemini_answer(q, final_context)
            truncated_context = "\n".join(final_context.split('\n')[:4])
            answers_with_context.append({"answer": answer.strip(), "context": truncated_context})

        except Exception as e:
            answers_with_context.append({"answer": f"Error: {e}", "context": ""})

    return {"answers": answers_with_context}
