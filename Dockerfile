FROM python:3.11-slim

WORKDIR /app

# Install essential system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only FIRST to save ~2GB RAM/Disk footprint
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and supervisor configuration
COPY backend/ ./backend
COPY frontend/ ./frontend
COPY supervisord.conf /app/supervisord.conf

# Environment variables for memory management and internal routing
ENV BACKEND_URL="http://127.0.0.1:8000/api/v1"
ENV CHROMA_PERSIST_DIR="/app/chroma_db"
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

EXPOSE 7860

CMD ["/usr/bin/supervisord", "-c", "/app/supervisord.conf"]