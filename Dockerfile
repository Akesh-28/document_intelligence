FROM python:3.11-slim

WORKDIR /app

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and supervisor configuration
COPY backend/ ./backend
COPY frontend/ ./frontend
COPY supervisord.conf /app/supervisord.conf

ENV BACKEND_URL="http://127.0.0.1:8000/api/v1"
ENV CHROMA_PERSIST_DIR="/app/chroma_db"

EXPOSE 7860

CMD ["/usr/bin/supervisord", "-c", "/app/supervisord.conf"]