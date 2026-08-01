---
title: Intelligent Document Engine
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---
# Intelligent Document Understanding Platform

A hybrid-retrieval RAG platform for ingesting, indexing, and querying documents (PDF, TXT, Markdown, CSV) with full source citation, cross-encoder re-ranking, and end-to-end execution telemetry.

---

## 1. Overview

This platform lets a user upload documents through a Streamlit interface, which are parsed, chunked, embedded, and persisted into a local ChromaDB vector store by a FastAPI backend. Queries are answered through a hybrid retrieval pipeline — dense vector search (ChromaDB, cosine similarity) fused with sparse BM25 keyword search via Reciprocal Rank Fusion, then refined by a cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`) — before being passed to a Groq-hosted LLM (`llama-3.3-70b-versatile`) for grounded, citation-bearing answer generation. Every query returns not just an answer, but the exact source chunks (with page/row citations and confidence scores) and a full latency/cost breakdown across every retrieval and generation stage.

Key capabilities:
- Multi-format ingestion: `.pdf`, `.txt`, `.md`, `.csv`
- SHA-256 content-hash deduplication (re-uploading identical content replaces the prior index entry)
- Hybrid dense + sparse retrieval with RRF fusion and cross-encoder re-ranking
- Grounded LLM answers with inline `[File, Page/Row]` citations
- Full document lifecycle management (upload, list, delete) from the UI or API
- Per-query telemetry: retrieval stage breakdown, LLM latency, token counts, estimated cost

---

## 2. Project Structure

```
document_intelligence/
├── backend/
│   ├── main.py                     # FastAPI app: registers all 4 REST endpoints directly
│   ├── config.py                   # Centralized Settings (Pydantic BaseSettings, loads .env)
│   ├── api/
│   │   └── routes/
│   │       ├── documents.py        # APIRouter: upload / list / delete document endpoints
│   │       └── query.py            # APIRouter: RAG query endpoint
│   ├── models/
│   │   └── schemas.py              # All Pydantic request/response schemas
│   └── services/
│       ├── ingestion.py            # File validation, hashing, parsing, chunking
│       ├── vector_store.py         # ChromaDB + BM25 dual-index service (VectorStoreService)
│       ├── reranker.py             # RRF fusion + cross-encoder re-ranking (RerankerService)
│       └── rag.py                  # Full RAG orchestration + Groq LLM call (RAGService)
├── frontend/
│   └── app.py                      # Streamlit UI: upload, document manager, query, telemetry
├── tests/
│   ├── test_ingestion.py           # Upload/dedup/replace behavior (FastAPI TestClient)
│   └── test_reranker_normalization.py  # Reranker score-normalization unit tests
├── chroma_db/                      # Persisted ChromaDB collection (created at runtime)
├── requirements.txt                # Python dependencies
├── .env                            # Local environment configuration (not committed)
├── .env.example                    # Environment variable template
└── .gitignore
```

---

## 3. Installation & Setup

### 3.1 Prerequisites
- Python 3.10+ (the bundled environment in this repository was built against CPython 3.14)
- A [Groq API key](https://console.groq.com/) for LLM generation (optional — the platform runs in retrieval-only mode without one)
- An OpenAI API key (optional — enables OpenAI `text-embedding-3-small` embeddings; without it, the platform automatically falls back to the local `all-MiniLM-L6-v2` SentenceTransformer model)

### 3.2 Create and Activate a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3.3 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.4 Configure Environment Variables
Copy the provided template and fill in your credentials:
```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
MAX_FILE_SIZE_MB=10
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHROMA_DB_DIR=./chroma_db
DEFAULT_TOP_K=4
```

> To enable OpenAI embeddings instead of the local SentenceTransformer fallback, also set `OPENAI_API_KEY` in `.env` — `backend/services/vector_store.py` checks for this variable directly at startup.

---

## 4. Running the Platform

The backend and frontend are run as two separate processes.

### 4.1 Start the FastAPI Backend
From the project root:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`, with interactive Swagger docs at `http://localhost:8000/docs`.

### 4.2 Start the Streamlit Frontend
In a second terminal (with the same virtual environment activated):
```bash
streamlit run frontend/app.py
```
The UI will open at `http://localhost:8501` and communicates with the backend at `http://localhost:8000/api/v1` (hardcoded `API_URL` in `frontend/app.py`) — ensure the backend is running first.

### 4.3 Running Tests
```bash
pytest tests/
```

---

## 5. API Reference

Base path: `/api/v1`

| Method | Endpoint | Summary | Request | Success Response |
|---|---|---|---|---|
| `POST` | `/documents/upload` | Upload and index a document (`.pdf`, `.txt`, `.md`, `.csv`) | `multipart/form-data`, field `file` | `200` — `UploadResponse` (document metadata, `replaced_doc_id` if a duplicate was replaced) |
| `GET` | `/documents` | List all indexed documents | — | `200` — array of `{doc_id, file_name, total_pages, chunk_count}` |
| `DELETE` | `/documents/{doc_id}` | Delete all vectors/chunks for a document | Path param: `doc_id` | `200` — confirmation message; `404` if not found |
| `POST` | `/query` | Run the hybrid retrieval + re-ranking + LLM generation pipeline | JSON: `{"prompt": str (min 3 chars), "top_k": int (1-10, default 4)}` | `200` — `QueryResponse` (`answer`, `citations[]`, `metrics`) |

For full request/response schemas, see the auto-generated OpenAPI documentation at `/docs` once the backend is running, or `backend/models/schemas.py` directly.