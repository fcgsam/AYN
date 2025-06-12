# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies (Tesseract + OpenCV for OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create the media folder for volume mount
RUN mkdir -p /app/media/vectorstore

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Start the app with Gunicorn and increased timeout
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "120", "AYNPRO.wsgi:application"]
