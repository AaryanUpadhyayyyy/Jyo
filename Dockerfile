# Use a specific Python 3.9 version to ensure compatibility.
# This should dictate the Python version used within the container.
FROM python:3.9-slim-buster

# Install build tools and SWIG (required for faiss-cpu compilation if no wheel is found)
# `build-essential` provides C/C++ compilers (gcc, g++) and make.
# `swig` is a tool for generating Python bindings for C/C++ libraries like FAISS.
RUN apt-get update && \
    apt-get install -y build-essential swig && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first and install dependencies.
# This leverages Docker's build cache: if requirements.txt doesn't change,
# this step (and its potentially long installation) won't be re-run on subsequent builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Render automatically sets the PORT environment variable for web services.
# Your FastAPI application should bind to this port.
# No EXPOSE instruction is strictly necessary as Render handles port mapping.

# Command to run your FastAPI application using Uvicorn.
# `--host 0.0.0.0` makes the app accessible from outside the container.
# `--port $PORT` uses the environment variable provided by Render.
# 'main:app' assumes your FastAPI instance is named 'app' in 'main.py'.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
