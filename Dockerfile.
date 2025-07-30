# Use a lightweight Python base image
FROM python:3.9-slim-buster

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
