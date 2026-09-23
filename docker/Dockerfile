# =============================================================================
# Unified Multi-Target Dockerfile
# Shared base layer for API service, GPU embedding service, and Ingestion worker
# =============================================================================

# --- Stage 1: Base Runtime Environment ---
FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements-lock.txt pyproject.toml /app/
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-lock.txt && \
    pip install --no-cache-dir -e .

# Copy application source and configs
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY migrations/ /app/migrations/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser
USER appuser


# --- Target 1: API Serving Service ---
FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]


# --- Target 2: GPU Embedding Service ---
FROM base AS embedding
EXPOSE 8001
ENV EMBEDDING_HOST=0.0.0.0 \
    EMBEDDING_PORT=8001
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1
CMD ["uvicorn", "src.embedding.service:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]


# --- Target 3: Ingestion Worker Service ---
FROM base AS worker
ENV RAW_INPUT_DIR=/data/raw \
    CHUNKS_OUTPUT_DIR=/data/chunks \
    INGESTION_WATCH_INTERVAL=10
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -m src.ingestion.worker --health-check || exit 1
CMD ["python", "-m", "src.ingestion.worker"]

