# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Install build tools and SWIG (required for faiss-cpu compilation)
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
# We don't need an explicit EXPOSE instruction if we're using Render's PORT.

# Command to run your FastAPI application using Uvicorn
# It's crucial to bind to 0.0.0.0 and the PORT environment variable.
# 'main:app' assumes your FastAPI instance is named 'app' in 'main.py'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
