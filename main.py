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
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Gemini API Setup with Key Rotation ---
# Read multiple API keys from a single environment variable
GEMINI_API_KEYS_STR = os.getenv("GEMINI_API_KEYS")
if not GEMINI_API_KEYS_STR:
    logger.error("GEMINI_API_KEYS environment variable not set.")
    raise ValueError("GEMINI_API_KEYS environment variable not set. Cannot proceed.")

# Use a deque for efficient key rotation
GEMINI_API_KEYS = deque(GEMINI_API_KEYS_STR.split(','))

# --- Feature Flags ---
ENABLE_LLM_RERANKING = os.getenv("ENABLE_LLM_RERANKING", "true").lower() == "true"
logger.info(f"Feature Flag: ENABLE_LLM_RERANKING is set to {ENABLE_LLM_RERANKING}")

def get_gemini_embedding(text: str, retries: int = 3, backoff_factor: float = 0.5) -> List[float]:
    """Generates Gemini embeddings with exponential backoff and key rotation."""
    for i in range(retries * len(GEMINI_API_KEYS)):
        current_key = GEMINI_API_KEYS[0]
        genai.configure(api_key=current_key)
        
        try:
            resp = genai.embed_content(model="models/embedding-001", content=[text])
            embedding_vector = resp["embedding"]
            if isinstance(embedding_vector, list) and len(embedding_vector) == 1 and isinstance(embedding_vector[0], list):
                embedding_vector = embedding_vector[0]
            
            # If successful, move the key to the back of the queue
            GEMINI_API_KEYS.rotate(-1)
            return embedding_vector
        except Exception as e:
            logger.warning(f"Embedding failed with key ending in {current_key[-5:]} (Attempt {i+1}/{retries*len(GEMINI_API_KEYS)}): {e}")
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                logger.warning("Quota exceeded. Rotating to next key.")
                GEMINI_API_KEYS.rotate(-1) # Rotate key on quota error
            
            if i < (retries * len(GEMINI_API_KEYS)) - 1:
                sleep_time = backoff_factor * (2 ** (i % retries))
                logger.info(f"Retrying after {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Final embedding generation failed after all retries: {e}")
                raise

def re_rank_chunks_with_llm(query: str, chunks: List[str], top_n_rerank: int = 5) -> List[str]:
    """Uses Gemini to re-rank a list of chunks based on their relevance to the query with key rotation."""
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
    
    for i in range(3 * len(GEMINI_API_KEYS)):
        current_key = GEMINI_API_KEYS[0]
        genai.configure(api_key=current_key)
        model = genai.GenerativeModel(model_name)
        
        try:
            logger.info(f"Calling Gemini for re-ranking with key ending in {current_key[-5:]}.")
            resp = model.generate_content(rerank_prompt)
            ranked_segments_text = resp.text.strip()
            
            re_ranked_chunks = [line.strip() for line in ranked_segments_text.split('\n') if line.strip()]
            final_re_ranked = [original for original in chunks if original.strip() in re_ranked_chunks][:top_n_rerank]
            
            if len(final_re_ranked) < top_n_rerank and len(chunks) > 0:
                logger.warning(f"Re-ranking returned fewer than {top_n_rerank} chunks. Falling back to original top N.")
            
            GEMINI_API_KEYS.rotate(-1)
            return final_re_ranked if final_re_ranked else chunks[:top_n_rerank]

        except Exception as e:
            logger.warning(f"Re-ranking failed with key ending in {current_key[-5:]} (Attempt {i+1}/{3*len(GEMINI_API_KEYS)}): {e}")
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                logger.warning("Quota exceeded. Rotating to next key.")
                GEMINI_API_KEYS.rotate(-1)
            
            sleep_time = 0.5 * (2 ** (i % 3))
            logger.info(f"Retrying after {sleep_time} seconds...")
            time.sleep(sleep_time)

    logger.error("Final re-ranking failed after all retries. Returning original chunks.")
    return chunks[:top_n_rerank]

def summarize_context(context: str) -> str:
    """Uses Gemini to provide a concise, 3-4 line summary of the context with key rotation."""
    for i in range(3 * len(GEMINI_API_KEYS)):
        current_key = GEMINI_API_KEYS[0]
        genai.configure(api_key=current_key)
        model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
        
        summary_prompt = (
            f"Given the following text, provide a concise summary of the key points in 3-4 lines.\n\n"
            f"**Text:**\n{context}"
        )
        
        try:
            logger.info(f"Calling Gemini for summarization with key ending in {current_key[-5:]}.")
            resp = model.generate_content(summary_prompt)
            GEMINI_API_KEYS.rotate(-1)
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"Summarization failed with key ending in {current_key[-5:]} (Attempt {i+1}/{3*len(GEMINI_API_KEYS)}): {e}")
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                logger.warning("Quota exceeded. Rotating to next key.")
                GEMINI_API_KEYS.rotate(-1)
            
            sleep_time = 0.5 * (2 ** (i % 3))
            logger.info(f"Retrying after {sleep_time} seconds...")
            time.sleep(sleep_time)
            
    logger.error("Final summarization failed after all retries. Returning default error.")
    return "Failed to summarize context."

def gemini_answer(question: str, context: str) -> str:
    """Uses Gemini to answer a question based on provided context with key rotation."""
    for i in range(3 * len(GEMINI_API_KEYS)):
        current_key = GEMINI_API_KEYS[0]
        genai.configure(api_key=current_key)
        model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
        
        prompt = (
            f"Given the following context from a policy/contract:\n\n{context}\n\n"
            f"Answer the question: '{question}' concisely and directly in a complete sentence. "
            f"If the answer is not in the context, say 'The provided text does not contain this information.'. "
            f"Do not add any additional commentary, reasoning, or quotes unless they are the direct answer."
        )
        
        try:
            resp = model.generate_content(prompt)
            GEMINI_API_KEYS.rotate(-1)
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"Answer failed with key ending in {current_key[-5:]} (Attempt {i+1}/{3*len(GEMINI_API_KEYS)}): {e}")
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                logger.warning("Quota exceeded. Rotating to next key.")
                GEMINI_API_KEYS.rotate(-1)
            
            sleep_time = 0.5 * (2 ** (i % 3))
            logger.info(f"Retrying after {sleep_time} seconds...")
            time.sleep(sleep_time)

    logger.error("Final answer generation failed after all retries.")
    return f"Error processing question: Failed to get an answer after multiple attempts due to API issues."

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

# --- Chunking ---
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
                sub_start_index += max_chunk_words - chunk_overlap_wo
