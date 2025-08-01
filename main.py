import os
import io
import requests
import fitz  # PyMuPDF
import docx
import faiss
import numpy as np
import logging  # Import logging module
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import google.generativeai as genai

# Configure logging for better visibility in Render logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Gemini API Setup ---
# *** IMPORTANT: Get API key from environment variable for security ***
# This ensures your API key is not hardcoded and is managed securely by Render.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set. Please set it in Render dashboard for production.")
    # For local testing, you might temporarily hardcode it here,
    # but REMOVE THIS LINE BEFORE DEPLOYING TO GITHUB/RENDER!
    # Example for local development ONLY: GEMINI_API_KEY = "YOUR_HARDCODED_API_KEY_HERE"
    raise ValueError("GEMINI_API_KEY environment variable not set. Cannot proceed.")

# --- CRITICAL DEBUG LOGGING FOR API KEY ---
# This line will print the first 5 characters of your API key to the logs.
# This is for DEBUGGING ONLY to confirm which key Render is using.
# REMOVE THIS LINE IN PRODUCTION FOR SECURITY!
if GEMINI_API_KEY:
    logger.info(f"DEBUG: Using Gemini API Key starting with: {GEMINI_API_KEY[:5]}*****")
# --- END DEBUG LOGGING ---

genai.configure(api_key=GEMINI_API_KEY)

# --- Feature Flags ---
# Control LLM-aided re-ranking via an environment variable for A/B testing
ENABLE_LLM_RERANKING = os.getenv("ENABLE_LLM_RERANKING", "true").lower() == "true"
logger.info(f"Feature Flag: ENABLE_LLM_RERANKING is set to {ENABLE_LLM_RERANKING}")


def get_gemini_embedding(text: str, retries: int = 3, backoff_factor: float = 0.5) -> List[float]:
    """
    Generates Gemini embeddings for the given text with exponential backoff.
    """
    for i in range(retries):
        try:
            # Call genai.embed_content directly with the model name and content.
            # The content is provided as a list, as expected by the API.
            resp = genai.embed_content(model="models/embedding-001", content=[text])
            
            # The response structure for resp["embedding"] can sometimes be a list containing
            # the actual embedding vector (e.g., [[...]]).
            # This block ensures we always extract the flat 768-dimensional list of floats.
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

def re_rank_chunks_with_llm(query: str, chunks: List[str], top_n_rerank: int = 5) -> List[str]:
    """
    Uses Gemini to re-rank a list of chunks based on their relevance to the query.
    This helps filter out less relevant chunks even if their embeddings were close.
    
    Args:
        query (str): The user's question.
        chunks (List[str]): A list of candidate text chunks.
        top_n_rerank (int): The number of top-ranked chunks to return.

    Returns:
        List[str]: A list of re-ranked (and potentially filtered) chunks.
    """
    if not chunks:
        return []

    model = genai.GenerativeModel("gemini-1.5-flash")

    rerank_prompt = (
        f"Given the following user query and a list of text segments, "
        f"identify the {top_n_rerank} most relevant segments that are most likely to contain the answer to the query. "
        f"Rank them from most relevant to least relevant.\n"
        f"Return ONLY the content of the selected segments, each on a new line, exactly as they appear in the input list.\n"
        f"Do NOT add any introductory or concluding remarks, just the segments.\n"
        f"If fewer than {top_n_rerank} relevant segments are found, return all relevant ones.\n\n"
        f"**User Query:** '{query}'\n\n"
        f"**Text Segments (each prefixed with 'Segment X:'):**\n"
    )
    for i, chunk in enumerate(chunks):
        rerank_prompt += f"Segment {i+1}: {chunk}\n"
    
    try:
        logger.info(f"Calling Gemini for chunk re-ranking with {len(chunks)} chunks.")
        resp = model.generate_content(rerank_prompt)
        ranked_segments_text = resp.text.strip()
        
        # Parse the response to extract the re-ranked chunks
        re_ranked_chunks = []
        for line in ranked_segments_text.split('\n'):
            # Simple heuristic: try to match the exact segment content
            # This can be improved with more robust parsing if Gemini's output varies
            for original_chunk in chunks:
                if original_chunk.strip() == line.strip():
                    re_ranked_chunks.append(original_chunk)
                    break
        
        # Ensure we return at most top_n_rerank unique chunks in order
        seen_chunks = set()
        final_re_ranked = []
        for chunk in re_ranked_chunks:
            if chunk not in seen_chunks:
                final_re_ranked.append(chunk)
                seen_chunks.add(chunk)
            if len(final_re_ranked) >= top_n_rerank:
                break
        
        # If Gemini didn't return enough, fall back to top_n_rerank from original list
        if len(final_re_ranked) < top_n_rerank and len(chunks) > 0:
            logger.warning(f"Gemini re-ranking returned fewer than {top_n_rerank} chunks. Falling back to top {top_n_rerank} from original retrieval.")
            return chunks[:top_n_rerank] # Fallback to original top N
        
        return final_re_ranked

    except Exception as e:
        logger.error(f"Error during LLM-aided chunk re-ranking for query '{query[:50]}...': {e}")
        # Fallback to original chunks if re-ranking fails
        return chunks[:top_n_rerank] # Return original top_n_rerank chunks if re-ranking fails


def gemini_answer(question: str, context: str) -> str:
    """
    Uses Gemini to answer a question based on provided context.
    Model changed from 'gemini-pro' to 'gemini-1.5-flash' for broader availability.
    This version simplifies the prompt for a concise, direct answer.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # --- SIMPLIFIED, CONCISE PROMPT FOR DIRECT ANSWERS ---
    prompt = (
        f"Given the following context from a policy/contract:\n\n{context}\n\n"
        f"Answer the question: '{question}' concisely and directly in a complete sentence. "
        f"If the answer is not in the context, say 'The provided text does not contain this information.'. "
        f"Do not add any additional commentary, reasoning, or quotes unless they are the direct answer."
    )
    # --- END SIMPLIFIED, CONCISE PROMPT ---

    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        logger.error(f"Error generating answer from Gemini for question: '{question[:50]}...': {e}")
        raise # Re-raise the exception after logging

# --- Document Parsing Functions ---
def extract_text_from_pdf(url: str) -> str:
    """Extracts text from a PDF document given its URL."""
    response = requests.get(url)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    doc = fitz.open(stream=response.content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()  # Close the document after processing
    return text

def extract_text_from_docx(url: str) -> str:
    """Extracts text from a DOCX document given its URL."""
    response = requests.get(url)
    response.raise_for_status()
    doc = docx.Document(io.BytesIO(response.content))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_email(url: str) -> str:
    """Extracts text from an EML (email) file given its URL."""
    response = requests.get(url)
    response.raise_for_status()
    from email import message_from_bytes  # Import here to avoid global import if not always used
    msg = message_from_bytes(response.content)
    payload = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            # Only consider plain text parts that are not attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                payload = part.get_payload(decode=True).decode(errors="ignore")
                break  # Take the first plain text part
    else:
        payload = msg.get_payload(decode=True).decode(errors="ignore")
    return payload

def extract_text(url: str) -> str:
    """Determines file type from URL and extracts text."""
    logger.info(f"Attempting to extract text from URL: {url}")
    try:
        if url.lower().endswith(".pdf"):
            return extract_text_from_pdf(url)
        elif url.lower().endswith(".docx"):
            return extract_text_from_docx(url)
        elif url.lower().endswith(".eml"):
            return extract_text_from_email(url)
        else:
            # Fallback/default logic for URLs without clear extension
            logger.warning(f"URL does not have a clear file extension: {url}. Attempting as PDF by default.")
            return extract_text_from_pdf(url)  # Default to PDF
    except requests.exceptions.RequestException as req_e:
        logger.error(f"Network or HTTP error fetching document from {url}: {req_e}")
        raise ValueError(f"Failed to fetch document from URL: {req_e}")
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        raise ValueError(f"Could not parse document. Check URL and file type: {e}")

# --- Chunking ---
def chunk_text(text: str, max_chunk_words: int = 1000, chunk_overlap_words: int = 150) -> List[str]:
    """
    Splits text into chunks, prioritizing paragraph boundaries, then word-based if paragraphs are too long.
    Includes overlap for context preservation.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk_words = []
    
    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        
        # Check if adding the current paragraph would exceed the max_chunk_words
        # and if there's already content in current_chunk_words
        if len(current_chunk_words) + len(paragraph_words) > max_chunk_words and current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
            
            # Create overlap for the next chunk
            overlap_words = current_chunk_words[max(0, len(current_chunk_words) - chunk_overlap_words):]
            current_chunk_words = overlap_words
        
        # Handle paragraphs that are larger than max_chunk_words by splitting them
        if len(paragraph_words) > max_chunk_words:
            # Add any accumulated current_chunk_words before processing large paragraph
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = [] # Reset for next accumulation

            # Split the large paragraph into sub-chunks with overlap
            sub_start_index = 0
            while sub_start_index < len(paragraph_words):
                sub_end_index = min(sub_start_index + max_chunk_words, len(paragraph_words))
                sub_chunk = " ".join(paragraph_words[sub_start_index:sub_end_index])
                chunks.append(sub_chunk)
                sub_start_index += max_chunk_words - chunk_overlap_words
                # Ensure sub_start_index doesn't go backwards if overlap is larger than chunk_size
                if sub_start_index < 0: 
                    sub_start_index = 0
            
            current_chunk_words = [] # Reset after handling large paragraph
        else:
            current_chunk_words.extend(paragraph_words)
    
    # Add any remaining words in current_chunk_words as the last chunk
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks

def keyword_search(query: str, all_chunks: List[str], top_n_keywords: int = 3) -> List[str]:
    """
    Performs a simple keyword search to find chunks containing query terms.
    Extracts keywords from the query and finds chunks that contain them.
    """
    query_words = [word.lower() for word in query.split() if len(word) > 2] # Filter out very short words
    
    relevant_keyword_chunks = []
    for chunk in all_chunks:
        if any(keyword in chunk.lower() for keyword in query_words):
            relevant_keyword_chunks.append(chunk)
            if len(relevant_keyword_chunks) >= top_n_keywords: # Limit number of keyword chunks
                break
    return relevant_keyword_chunks

# --- FAISS Vector Store ---
class VectorStore:
    """In-memory FAISS vector store for document chunks."""
    def __init__(self, dim: int):
        # Initialize FAISS index with the correct dimension
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []

    def add(self, embeddings: List[List[float]], chunks: List[str]):
        """Adds embeddings and their corresponding text chunks to the store."""
        # Ensure embeddings are float32 as required by FAISS
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunks.extend(chunks)

    def search(self, embedding: List[float], top_k: int = 8): # Increased top_k to 8 for more context
        """Searches for the most similar chunks to a given embedding."""
        # Ensure query embedding is float32 and 2D array for FAISS search
        D, I = self.index.search(np.array([embedding]).astype("float32"), top_k)
        return [self.chunks[i] for i in I[0]]

# --- FastAPI Setup ---
app = FastAPI(
    title="LLM-Powered Document Q&A System",
    description="API for extracting text from documents, chunking, embedding, and answering questions using Gemini.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """Root endpoint to confirm the API is running."""
    return {"message": "LLM-Powered Query Retrieval System is running!", "endpoint": "/api/v1/hackrx/run"}

class QueryRequest(BaseModel):
    documents: str  # URL to the document (e.g., PDF, DOCX, EML)
    questions: List[str]  # List of questions to ask about the document

class QueryResponse(BaseModel):
    answers: List[str]  # List of answers corresponding to the questions

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
def run_query(req: QueryRequest):
    """
    Main endpoint to process a document and answer questions.
    Expects a document URL and a list of questions.
    """
    logger.info(f"Received request for document: {req.documents} with {len(req.questions)} questions.")

    # 1. Download and parse document
    text = ""
    try:
        text = extract_text(req.documents)
        logger.info(f"Successfully extracted {len(text)} characters from document: {req.documents}")
        if not text:
            raise HTTPException(status_code=400, detail="Extracted text is empty. Document might be unparseable or empty.")
    except ValueError as ve:
        logger.error(f"Document parsing failed: {ve}")
        raise HTTPException(status_code=400, detail=f"Document parsing failed: {ve}")
    except HTTPException:  # Re-raise if it was already an HTTPException
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during document parsing: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during document parsing: {e}")

    # 2. Chunk and embed
    chunks = []
    embeddings = []
    store = None
    try:
        # Calling chunk_text with new parameters for larger, overlapping, paragraph-aware chunks
        chunks = chunk_text(text, max_chunk_words=1000, chunk_overlap_words=150) 
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from document. Text might be too short or chunking failed.")
        logger.info(f"Created {len(chunks)} chunks from document.")

        embeddings = [get_gemini_embedding(chunk) for chunk in chunks]
        logger.info(f"Generated {len(embeddings)} embeddings for chunks.")

        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate any embeddings.")

        # --- CRITICAL DEBUG LOGGING ---
        # These logs help diagnose the structure of the embeddings
        if len(embeddings) > 0:
            logger.info(f"DEBUG: Type of embeddings list: {type(embeddings)}")
            logger.info(f"DEBUG: Type of first embedding (embeddings[0]): {type(embeddings[0])}")
            logger.info(f"DEBUG: Length of first embedding (len(embeddings[0])): {len(embeddings[0])}")
            # Log a small snippet of the first embedding to see its structure
            logger.info(f"DEBUG: First 10 elements of embeddings[0]: {embeddings[0][:10]}")
        # --- END CRITICAL DEBUG LOGGING ---

        # --- Embedding Dimension Consistency Check ---
        # This ensures all embeddings have the same expected dimension (768 for embedding-001)
        if embeddings:
            first_embedding_dim = len(embeddings[0])
            # The embedding-001 model is expected to return 768 dimensions.
            # If it's not 768, it indicates an unexpected issue with the model response.
            if first_embedding_dim != 768:
                logger.error(f"Expected embedding dimension 768, but got {first_embedding_dim}. This is unexpected.")
                # You might choose to raise an error here or proceed if you want to allow other dimensions
                # For now, we proceed as FAISS will handle the dimension passed to VectorStore(dim)
            logger.info(f"First embedding dimension: {first_embedding_dim}")
            for i, emb in enumerate(embeddings):
                if len(emb) != first_embedding_dim:
                    logger.error(f"Inconsistent embedding dimension at index {i}. Expected {first_embedding_dim}, got {len(emb)}")
                    raise HTTPException(status_code=500, detail="Inconsistent embedding dimensions generated.")
            dim = first_embedding_dim
        else:
            raise HTTPException(status_code=500, detail="No embeddings generated, cannot determine dimension.")
        # --- END Embedding Dimension Consistency Check ---

        logger.info(f"Initializing VectorStore with dimension: {dim}")
        store = VectorStore(dim)
        store.add(embeddings, chunks)
        logger.info("Successfully created and populated FAISS vector store.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during chunking, embedding, or vector store creation: {e}")
        # Re-raise with a more specific detail if it's the unpacking error
        if "too many values to unpack" in str(e):
            raise HTTPException(status_code=500, detail=f"Processing error (chunking/embedding/vector store): Possible dimension mismatch or FAISS issue: {e}")
        else:
            raise HTTPException(status_code=500, detail=f"Processing error (chunking/embedding/vector store): {e}")

    # 3. For each question: retrieve, reason, answer
    answers = []
    for i, q in enumerate(req.questions):
        question_logger = logger.getChild(f"Question-{i+1}")  # Specific logger for each question
        try:
            question_logger.info(f"Processing question {i+1}/{len(req.questions)}: '{q[:100]}...'")
            q_emb = get_gemini_embedding(q)
            logger.info(f"Query embedding dimension: {len(q_emb)}")
            
            # --- Retrieval Augmentation ---
            # 1. Semantic Search (FAISS)
            faiss_relevant_chunks = store.search(q_emb, top_k=8) # Initial top_k for FAISS
            question_logger.info(f"FAISS retrieved {len(faiss_relevant_chunks)} chunks.")

            # 2. Keyword Search Augmentation
            keyword_relevant_chunks = keyword_search(q, chunks, top_n_keywords=3)
            question_logger.info(f"Keyword search retrieved {len(keyword_relevant_chunks)} chunks.")

            # Combine and deduplicate chunks
            combined_candidate_chunks = list(dict.fromkeys(faiss_relevant_chunks + keyword_relevant_chunks))
            question_logger.info(f"Combined candidate chunks (deduplicated): {len(combined_candidate_chunks)} chunks.")

            # 3. LLM-Aided Re-ranking (Conditional based on feature flag)
            if ENABLE_LLM_RERANKING:
                final_context_chunks = re_rank_chunks_with_llm(q, combined_candidate_chunks, top_n_rerank=5) # Re-rank to top 5
                question_logger.info(f"LLM re-ranked to {len(final_context_chunks)} final context chunks (re-ranking enabled).")
            else:
                final_context_chunks = combined_candidate_chunks[:5] # Fallback to top 5 from combined if re-ranking disabled
                question_logger.info(f"Re-ranking disabled. Using top 5 from combined candidate chunks.")
            # --- END Retrieval Augmentation ---

            # --- RIGOROUS LOGGING: Log the chunks passed to LLM ---
            question_logger.info(f"Final context chunks passed to LLM for Q{i+1}:")
            for j, chunk in enumerate(final_context_chunks):
                question_logger.info(f"  Chunk {j+1} (length {len(chunk.split())} words): '{chunk[:200]}...'")
            # --- END RIGOROUS LOGGING ---

            context = "\n---\n".join(final_context_chunks)
            
            # Answer using Gemini
            answer = gemini_answer(q, context)
            answers.append(answer.strip())
            question_logger.info(f"Successfully answered question {i+1}.")
        except Exception as e:
            question_logger.error(f"Error processing question {i+1}: {e}")
            answers.append(f"Error processing question: {e}")  # Return error message for specific question

    logger.info("All questions processed. Returning responses.")
    return {"answers": answers}
