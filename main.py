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
from openai import OpenAI # The OpenAI library is compatible with OpenRouter's API

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API Setup for Unified OpenRouter Access ---
# A single key is used to access both embedding and chat models via OpenRouter.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    logger.error("DEEPSEEK_API_KEY environment variable not set.")
    raise ValueError("DEEPSEEK_API_KEY environment variable not set. Cannot proceed.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=DEEPSEEK_API_KEY,
)
logger.info(f"DEBUG: Using API Key starting with: {DEEPSEEK_API_KEY[:5]}*****")

# --- Feature Flags ---
ENABLE_LLM_RERANKING = os.getenv("ENABLE_LLM_RERANKING", "true").lower() == "true"
logger.info(f"Feature Flag: ENABLE_LLM_RERANKING is set to {ENABLE_LLM_RERANKING}")

def get_embedding(text: str, model: str = "nomic-ai/nomic-embed-text-v1.5") -> List[float]:
    """Generates embeddings using a model from OpenRouter."""
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding with {model}: {e}")
        raise

def re_rank_chunks_with_llm(query: str, chunks: List[str], top_n_rerank: int = 5, retries: int = 3, backoff_factor: float = 0.5) -> List[str]:
    """
    Uses Deepseek to re-rank a list of chunks based on their relevance to the query.
    """
    if not chunks:
        return []

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
            logger.info(f"Calling Deepseek for chunk re-ranking (Attempt {i+1}/{retries}) with {len(chunks)} chunks.")
            chat_response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[{"role": "user", "content": rerank_prompt}],
            )
            ranked_segments_text = chat_response.choices[0].message.content.strip()
            
            re_ranked_chunks = [line.strip() for line in ranked_segments_text.split('\n') if line.strip()]
            final_re_ranked = [original for original in chunks if original.strip() in re_ranked_chunks][:top_n_rerank]
            
            if len(final_re_ranked) < top_n_rerank and len(chunks) > 0:
                logger.warning(f"Deepseek re-ranking returned fewer than {top_n_rerank} chunks. Falling back to original top N.")
                return chunks[:top_n_rerank]
            return final_re_ranked
        except Exception as e:
            logger.warning(f"Deepseek re-ranking failed (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                sleep_time = backoff_factor * (2 ** i)
                time.sleep(sleep_time)
            else:
                logger.error(f"Final Deepseek re-ranking failed after {retries} retries: {e}")
                return chunks[:top_n_rerank]

def summarize_context(context: str, retries: int = 3, backoff_factor: float = 0.5) -> str:
    """
    Uses Deepseek to provide a concise, 3-4 line summary of the context.
    """
    summary_prompt = (
        f"Given the following text, provide a concise summary of the key points in 3-4 lines.\n\n"
        f"**Text:**\n{context}"
    )
    for i in range(retries):
        try:
            logger.info("Calling Deepseek for context summarization.")
            chat_response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[{"role": "user", "content": summary_prompt}],
            )
            return chat_response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Deepseek summarization failed (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                sleep_time = backoff_factor * (2 ** i)
                time.sleep(sleep_time)
            else:
                logger.error(f"Final Deepseek summarization failed after {retries} retries: {e}")
                return "Failed to summarize context."

def deepseek_answer(question: str, context: str, retries: int = 3, backoff_factor: float = 0.5) -> str:
    """
    Uses Deepseek to answer a question based on provided context with exponential backoff.
    """
    prompt = (
        f"Given the following context from a policy/contract:\n\n{context}\n\n"
        f"Answer the question: '{question}' concisely and directly in a complete sentence. "
        f"If the answer is not in the context, say 'The provided text does not contain this information.'. "
        f"Do not add any additional commentary, reasoning, or quotes unless they are the direct answer."
    )
    
    for i in range(retries):
        try:
            chat_response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[{"role": "user", "content": prompt}],
            )
            return chat_response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Deepseek answer generation failed (Attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                sleep_time = backoff_factor * (2 ** i)
                time.sleep(sleep_time)
            else:
                logger.error(f"Final Deepseek answer generation failed after {retries} retries: {e}")
                return f"Error processing question: Failed to get an answer after multiple attempts due to API issues. Last error: {e}"

# --- Document Parsing and other functions remain the same ---
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
    logger.info(f"Attempting to extract text from URL: {url}")
    try:
        if url.lower().endswith(".pdf"):
            return extract_text_from_pdf(url)
        elif url.lower().endswith(".docx"):
            return extract_text_from_docx(url)
        elif url.lower().endswith(".eml"):
            return extract_text_from_email(url)
        else:
            logger.warning(f"URL does not have a clear file extension: {url}. Attempting as PDF by default.")
            return extract_text_from_pdf(url)
    except requests.exceptions.RequestException as req_e:
        logger.error(f"Network or HTTP error fetching document from {url}: {req_e}")
        raise ValueError(f"Failed to fetch document from URL: {req_e}")
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        raise ValueError(f"Could not parse document. Check URL and file type: {e}")

def chunk_text(text: str, max_chunk_words: int = 1000, chunk_overlap_words: int = 150) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk_words = []
    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        if len(current_chunk_words) + len(paragraph_words) > max_chunk_words and current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
            overlap_words = current_chunk_words[max(0, len(current_chunk_words) - chunk_overlap_words):]
            current_chunk_words = overlap_words
        if len(paragraph_words) > max_chunk_words:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []
            sub_start_index = 0
            while sub_start_index < len(paragraph_words):
                sub_end_index = min(sub_start_index + max_chunk_words, len(paragraph_words))
                sub_chunk = " ".join(paragraph_words[sub_start_index:sub_end_index])
                chunks.append(sub_chunk)
                sub_start_index += max_chunk_words - chunk_overlap_words
                if sub_start_index < 0:
                    sub_start_index = 0
            current_chunk_words = []
        else:
            current_chunk_words.extend(paragraph_words)
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
    return chunks

def keyword_search(query: str, all_chunks: List[str], top_n_keywords: int = 3) -> List[str]:
    query_words = [word.lower() for word in query.split() if len(word) > 2]
    relevant_keyword_chunks = []
    for chunk in all_chunks:
        if any(keyword in chunk.lower() for keyword in query_words):
            relevant_keyword_chunks.append(chunk)
            if len(relevant_keyword_chunks) >= top_n_keywords:
                break
    return relevant_keyword_chunks

# --- FAISS Vector Store ---
class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []
    def add(self, embeddings: List[List[float]], chunks: List[str]):
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunks.extend(chunks)
    def search(self, embedding: List[float], top_k: int = 8):
        D, I = self.index.search(np.array([embedding]).astype("float32"), top_k)
        return [self.chunks[i] for i in I[0]]

# --- FastAPI Setup ---
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
    text = ""
    try:
        text = extract_text(req.documents)
        logger.info(f"Successfully extracted {len(text)} characters from document: {req.documents}")
        if not text:
            raise HTTPException(status_code=400, detail="Extracted text is empty. Document might be unparseable or empty.")
    except ValueError as ve:
        logger.error(f"Document parsing failed: {ve}")
        raise HTTPException(status_code=400, detail=f"Document parsing failed: {ve}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during document parsing: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during document parsing: {e}")

    chunks = []
    embeddings = []
    store = None
    try:
        chunks = chunk_text(text, max_chunk_words=1000, chunk_overlap_words=150)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from document. Text might be too short or chunking failed.")
        logger.info(f"Created {len(chunks)} chunks from document.")
        embeddings = [get_embedding(chunk) for chunk in chunks]
        logger.info(f"Generated {len(embeddings)} embeddings for chunks.")
        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate any embeddings.")

        if embeddings:
            first_embedding_dim = len(embeddings[0])
            if first_embedding_dim != 1536: # OpenAI embeddings dimension
                logger.error(f"Expected embedding dimension 1536, but got {first_embedding_dim}. This is unexpected.")
            logger.info(f"First embedding dimension: {first_embedding_dim}")
            for i, emb in enumerate(embeddings):
                if len(emb) != first_embedding_dim:
                    logger.error(f"Inconsistent embedding dimension at index {i}. Expected {first_embedding_dim}, got {len(emb)}")
                    raise HTTPException(status_code=500, detail="Inconsistent embedding dimensions generated.")
            dim = first_embedding_dim
        else:
            raise HTTPException(status_code=500, detail="No embeddings generated, cannot determine dimension.")
        logger.info(f"Initializing VectorStore with dimension: {dim}")
        store = VectorStore(dim)
        store.add(embeddings, chunks)
        logger.info("Successfully created and populated FAISS vector store.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during chunking, embedding, or vector store creation: {e}")
        if "too many values to unpack" in str(e):
            raise HTTPException(status_code=500, detail=f"Processing error (chunking/embedding/vector store): Possible FAISS dimension issue: {e}")
        else:
            raise HTTPException(status_code=500, detail=f"Processing error (chunking/embedding/vector store): {e}")

    answers_with_context = []
    for i, q in enumerate(req.questions):
        question_logger = logger.getChild(f"Question-{i+1}")
        try:
            question_logger.info(f"Processing question {i+1}/{len(req.questions)}: '{q[:100]}...'")
            q_emb = get_embedding(q)
            logger.info(f"Query embedding dimension: {len(q_emb)}")

            faiss_relevant_chunks = store.search(q_emb, top_k=8)
            keyword_relevant_chunks = keyword_search(q, chunks, top_n_keywords=3)
            combined_candidate_chunks = list(dict.fromkeys(faiss_relevant_chunks + keyword_relevant_chunks))
            question_logger.info(f"Combined candidate chunks (deduplicated): {len(combined_candidate_chunks)} chunks.")

            if ENABLE_LLM_RERANKING:
                final_context_chunks = re_rank_chunks_with_llm(q, combined_candidate_chunks, top_n_rerank=5)
                question_logger.info(f"LLM re-ranked to {len(final_context_chunks)} final context chunks (re-ranking enabled).")
            else:
                final_context_chunks = combined_candidate_chunks[:5]
                question_logger.info(f"Re-ranking disabled. Using top 5 from combined candidate chunks.")

            question_logger.info(f"Final context chunks passed to LLM for Q{i+1}:")
            for j, chunk in enumerate(final_context_chunks):
                question_logger.info(f"  Chunk {j+1} (length {len(chunk.split())} words): '{chunk[:200]}...'")

            context_for_llm = "\n---\n".join(final_context_chunks)
            answer = deepseek_answer(q, context_for_llm)
            
            summarized_context = summarize_context(context_for_llm)
            
            answers_with_context.append({"answer": answer.strip(), "context": summarized_context})
            question_logger.info(f"Successfully answered question {i+1}.")
        except Exception as e:
            question_logger.error(f"Error processing question {i+1}: {e}")
            answers_with_context.append({"answer": f"Error processing question: {e}", "context": ""})

    logger.info("All questions processed. Returning responses.")
    return {"answers": answers_with_context}
#hey
