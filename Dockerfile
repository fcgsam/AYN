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

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Run Django migrations (if needed)
# RUN python manage.py migrate

# Start the app with Gunicorn (production-ready WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "AYNPRO.wsgi:application"]