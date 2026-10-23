# Multi-stage production Dockerfile
FROM python:3.10-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Install system dependencies for OpenCV and audio processing (libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement specifications and install Python wheels
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend, configs, models, and prebuilt frontend
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY models/ ./models/
COPY frontend/build/ ./frontend/build/

# Set working directory to backend
WORKDIR /app/backend

EXPOSE 5000

# Run with Gunicorn eventlet worker
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "app:app"]
