# Use a specific Python 3.9 version to ensure compatibility
FROM python:3.9-slim-buster

# Install build tools and SWIG (required for faiss-cpu compilation if no wheel is found)
# We need build-essential for C/C++ compilers and related tools
# We need swig for generating Python bindings for FAISS
RUN apt-get update && \
    apt-get install -y build-essential swig && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies first (for better Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Render automatically sets the PORT environment variable.
# Your app should bind to this port.

# Command to run your FastAPI application using Uvicorn
# It's crucial to bind to 0.0.0.0 and the PORT environment variable.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
