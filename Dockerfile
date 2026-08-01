FROM python:3.11-slim

# Install system dependencies required for building C extensions & supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directory for ChromaDB vector store persistence
RUN mkdir -p /app/chroma_db

# Default environment variables
ENV PORT=7860
ENV BACKEND_URL=http://127.0.0.1:8000/api/v1
ENV CHROMA_PERSIST_DIR=/app/chroma_db

EXPOSE 7860

# Healthcheck targeting backend liveness probe
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health || exit 1

# Start supervisor process manager
CMD ["supervisord", "-c", "/app/supervisord.conf"]