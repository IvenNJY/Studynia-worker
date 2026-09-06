# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for OCR and PDF parsing
RUN apt-get update -qq && \
    apt-get install -y -qq tesseract-ocr poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI code into the container
COPY . .

# Start Uvicorn when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]