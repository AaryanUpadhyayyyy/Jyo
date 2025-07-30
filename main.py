import os
import io
import requests
import fitz  # PyMuPDF
import docx
import faiss
import numpy as np
import logging  # Import logging module
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
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

def get_gemini_embedding(text: str) -> List[float]:
    """
    Generates Gemini embeddings for the given text.
    Includes a critical fix to ensure the embedding is a flat list of floats.
    """
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
            logger.info("DEBUG: Extracted inner list from embedding response.")
            
        return embedding_vector
        
    except Exception as e:
        logger.error(f"Error generating embedding for text: '{text[:50]}...': {e}")
        raise # Re-raise the exception after logging

def gemini_answer(question: str, context: str) -> str:
    """
    Uses Gemini to answer a question based on provided context.
    Model changed from 'gemini-pro' to 'gemini-1.5-flash' for broader availability.
    """
    # Changed model to 'gemini-1.5-flash' for better compatibility and availability
    model = genai.GenerativeModel("gemini-1.5-flash") 
    prompt = (
        f"Given the following context from a policy/contract:\n\n{context}\n\n"
        f"Answer the question: '{question}'\n"
        "If possible, cite the relevant clause(s) and explain your reasoning."
    )
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
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Splits text into smaller chunks for embedding."""
    words = text.split()
    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

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

    def search(self, embedding: List[float], top_k: int = 3):
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
        chunks = chunk_text(text)
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
            
            # Retrieve relevant chunks
            # Ensure top_k does not exceed the number of available chunks
            relevant_chunks = store.search(q_emb, top_k=min(3, len(chunks)))
            context = "\n---\n".join(relevant_chunks)
            
            # Answer using Gemini
            answer = gemini_answer(q, context)
            answers.append(answer.strip())
            question_logger.info(f"Successfully answered question {i+1}.")
        except Exception as e:
            question_logger.error(f"Error processing question {i+1}: {e}")
            answers.append(f"Error processing question: {e}")  # Return error message for specific question

    logger.info("All questions processed. Returning responses.")
    return {"answers": answers}

# --- For local development or Render deployment ---
# This block is typically removed or commented out for Docker-based deployments
# like on Render, as the Dockerfile's CMD instruction handles the server startup.
# if __name__ == "__main__":
#     import uvicorn
#     # Get port from environment variable set by Render, or default to 8000
#     port = int(os.environ.get("PORT", 8000))
#     logger.info(f"Starting Uvicorn server on http://0.0.0.0:{port}")
#     # reload=True is good for local development, set to False for production on Render
#     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True if os.getenv("ENV") == "development" else False)
