# Use a specific Python 3.9 version to ensure compatibility.
FROM python:3.9-slim-buster

# Install build tools. 'openai' and other packages might have C extensions that need compiling.
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Command to run your FastAPI application using Uvicorn.
# '--host 0.0.0.0' makes the app accessible from outside the container.
# '--port $PORT' uses the environment variable provided by Render.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
