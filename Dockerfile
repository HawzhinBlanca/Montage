# Multi-stage Docker build for AI Video Processing Pipeline

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set labels
LABEL org.opencontainers.image.title="AI Video Processing Pipeline"
LABEL org.opencontainers.image.description="Production-ready AI video processing system"
LABEL org.opencontainers.image.vendor="AI Video Pipeline Team"
LABEL org.opencontainers.image.source="https://github.com/your-org/ai-video-pipeline"

# Build arguments
ARG VERSION=latest
ARG COMMIT_SHA=unknown

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/home/appuser/.local/bin:$PATH"
ENV VERSION=$VERSION
ENV COMMIT_SHA=$COMMIT_SHA

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Create directories
RUN mkdir -p /app /tmp/video_processing && \
    chown -R appuser:appuser /app /tmp/video_processing

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 9099

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py"]