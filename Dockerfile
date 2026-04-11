# ============================================
# Dockerfile — OCR Receipt Extraction Tool
# ============================================
# Uses Python 3.11 slim base with Tesseract OCR installed
# Optimized for Render deployment
# ============================================

# Stage 1: Use Python 3.11 slim as the base image
# slim variant is smaller than full image but has enough for our needs
FROM python:3.11-slim

# ============================================
# Install system dependencies
# ============================================
# Tesseract OCR engine + language data + OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Set up the working directory
# ============================================
WORKDIR /app

# ============================================
# Install Python dependencies
# ============================================
# Copy requirements first for Docker layer caching —
# dependencies only reinstall when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# Copy the application code
# ============================================
COPY . .

# ============================================
# Create necessary directories
# ============================================
RUN mkdir -p outputs temp_uploads samples

# ============================================
# Set environment variables
# ============================================
# Prevents Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONUNBUFFERED=1
# Tesseract binary location (already in PATH from apt install)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# ============================================
# Expose the API port
# ============================================
# Render sets the PORT environment variable automatically
EXPOSE 8000

# ============================================
# Health check
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# ============================================
# Start the FastAPI server
# ============================================
# Use PORT env variable (Render sets this), default to 8000
# --host 0.0.0.0 makes the server accessible from outside the container
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
