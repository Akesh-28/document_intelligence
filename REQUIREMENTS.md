# REQUIREMENTS.md
## Intelligent Document Understanding & Retrieval Engine — Functional Specifications

**Document Status:** Derived directly from source inspection of the implemented codebase (FastAPI backend + Streamlit frontend + ChromaDB vector store).
**Version:** 1.0.0 (per `backend/main.py` FastAPI app metadata)

---

## 1. Document Scope & Features

### 1.1 Supported File Formats
The platform enforces a strict allow-list at two independent layers — the API entrypoint (`backend/main.py`, `backend/api/routes/documents.py`) and the ingestion service (`backend/services/ingestion.py::validate_file`) — both derived from `settings.ALLOWED_EXTENSIONS`:

| Extension | MIME Category | Parsing Strategy |
|---|---|---|
| `.pdf` | Portable Document Format | Page-by-page text extraction via `pypdf.PdfReader` |
| `.txt` | Plain text | Whole-file decode, treated as a single logical page |
| `.md` | Markdown | Whole-file decode, treated as a single logical page (no markdown-aware parsing) |
| `.csv` | Comma-separated values | Row-by-row structured parsing via `csv.DictReader` |

Any other extension is rejected with `HTTP 400 Bad Request` before file bytes are processed further.

### 1.2 Ingestion Features
- **SHA-256 content-hash deduplication.** Every upload is hashed (`hashlib.sha256`) prior to indexing. If a document with an identical hash already exists in the vector store, the pre-existing document's vectors are deleted and replaced by the new upload (same content, new `doc_id`).
- **Multi-encoding text decoding.** Plain text/Markdown files attempt `utf-8-sig` → `utf-8` → `latin-1` in order. CSV files additionally attempt `cp1252`. Decoding failure across all attempts returns `HTTP 400`.
- **File size enforcement.** Uploads exceeding `MAX_FILE_SIZE_MB` (default: 10 MB) are rejected with `HTTP 413 Request Entity Too Large`.
- **Empty-content rejection.** PDFs with no extractable text, empty text files, and CSVs that yield zero parseable rows are rejected with `HTTP 400`.
- **Recursive character-based chunking** for PDF/TXT/MD content via LangChain's `RecursiveCharacterTextSplitter` (chunk size 1000 chars, overlap 200 chars — see Section 3).
- **Row-level chunking for CSV**, where each row becomes one chunk formatted as `Row {n}: col1: val1 | col2: val2 | ...` (empty column/value pairs are dropped).

### 1.3 Metadata Fields
Each indexed chunk carries the following metadata into ChromaDB (`backend/services/ingestion.py`):

| Field | Type | Applies To | Description |
|---|---|---|---|
| `doc_id` | string (UUID4) | all | Unique identifier for the parent document |
| `content_hash` | string (SHA-256 hex) | all | Used for duplicate detection |
| `file_name` | string | all | Original uploaded filename |
| `chunk_id` | string | all | `{doc_id}_p{page}_c{chunk_index}` for text formats, `{doc_id}_r{row}` for CSV |
| `page_number` | int | pdf/txt/md | Source page (1-indexed); for CSV this is set equal to the row number |
| `row_number` | int | csv | Source row (1-indexed, header excluded) |
| `total_pages` | int | all | Total page count (text formats) or total row count (CSV) |
| `file_type` | string | all | `pdf`, `txt`, `md`, or `csv` |

### 1.4 Search / Query Capabilities
- **Hybrid retrieval** combining dense vector similarity (ChromaDB, cosine space) and sparse keyword search (BM25Okapi via `rank_bm25`), enabled by default (`ENABLE_HYBRID_SEARCH=True`).
- **Reciprocal Rank Fusion (RRF)** to merge the two candidate lists into a single ranked set.
- **Cross-encoder re-ranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the fused candidate set, enabled by default (`ENABLE_RERANKING=True`), producing a min-max normalized `relevance_score` (0.0–1.0) alongside the raw model logit.
- **Configurable Top-K** at query time (`top_k`, 1–10, default 4), applied as the final truncation after re-ranking.
- **Grounded LLM answer generation** via Groq (`langchain_groq.ChatGroq`, model `llama-3.3-70b-versatile`, temperature 0.0) constrained to the retrieved context, with inline citation instructions baked into the system prompt.
- **Per-query execution telemetry**: retrieval latency broken down by BM25, dense embedding, ChromaDB I/O, RRF fusion, and re-ranking stages; LLM latency; total latency; prompt/completion token counts; and an estimated USD cost.

### 1.5 Lifecycle Operations
| Operation | Endpoint | Description |
|---|---|---|
| Upload & Index | `POST /api/v1/documents/upload` | Validates, parses, chunks, embeds, and indexes a document. Replaces any existing document with identical content hash. |
| List | `GET /api/v1/documents` | Returns a de-duplicated summary of every indexed document (doc_id, file_name, total_pages, chunk_count), aggregated from chunk-level metadata. |
| Delete | `DELETE /api/v1/documents/{doc_id}` | Purges all chunks belonging to `doc_id` from ChromaDB and rebuilds the in-memory BM25 index. Returns `404` if `doc_id` is not found. |
| Query | `POST /api/v1/query` | Executes the full hybrid-search → fusion → re-rank → LLM-generation pipeline and returns a grounded answer with citations and telemetry. |

---

## 2. Functional Requirements Specification

### 2.1 Workflow: Document Upload
1. User selects a file via the Streamlit sidebar uploader (`frontend/app.py`) or calls the API directly.
2. Backend validates the presence of a filename and its extension against `{pdf, txt, md, csv}`.
3. Backend reads file bytes and computes a SHA-256 hash.
4. If a document with the same hash already exists in the index, its chunks are deleted (`vector_store_service.delete_document_by_id`).
5. `process_and_chunk_document()` dispatches to the format-specific extractor, producing `(doc_id, chunks, metadatas, total_pages)`.
6. Chunks and sanitized metadata are embedded and written to the ChromaDB collection; the in-memory BM25 index is rebuilt from the full corpus.
7. Response returns `UploadResponse` containing `document` (a `DocumentMetadata` object) and, if applicable, `replaced_doc_id`.

**API Contract — `POST /api/v1/documents/upload`**
- **Request:** `multipart/form-data`, field `file` (binary).
- **Constraints:** filename required; extension ∈ `{pdf, txt, md, csv}`; size ≤ `MAX_FILE_SIZE_MB` (10 MB default); content must yield ≥1 non-empty chunk.
- **Success Response (200):**
```json
{
  "message": "Successfully indexed document.",
  "document": {
    "doc_id": "uuid4-string",
    "file_name": "example.pdf",
    "file_type": ".pdf",
    "page_count": 12,
    "total_chunks": 34,
    "upload_timestamp": "2026-01-01T00:00:00+00:00"
  },
  "replaced_doc_id": null
}
```
- **Error Responses:** `400` (invalid filename, unsupported extension, unreadable/empty content), `413` (file too large), `500` (unexpected processing failure).

### 2.2 Workflow: Query / Retrieval
1. User submits a natural-language question (Streamlit main panel or direct API call), with an adjustable Top-K slider (1–10) in the UI.
2. Backend validates that the prompt is non-empty (`min_length=3` at the schema level; the legacy handler in `main.py` additionally rejects whitespace-only prompts).
3. Dense retrieval fetches `DENSE_TOP_K` (10) candidates from ChromaDB; sparse BM25 retrieval fetches `SPARSE_TOP_K` (10) candidates — both run when hybrid search is enabled.
4. Candidates are merged via Reciprocal Rank Fusion (`k=60`).
5. The fused candidate set is passed through the cross-encoder re-ranker, which returns the top `top_k` chunks sorted by raw logit, annotated with a normalized `relevance_score`.
6. Retrieved chunk text is cleaned of encoding artifacts (`ftfy.fix_text` plus explicit mojibake replacements) and assembled into a citation-tagged context block (`[File: name, Page: X]` or `[File: name, Row: Y]`).
7. The context and question are passed to the Groq LLM chain (`ChatPromptTemplate` → `ChatGroq`) under a system prompt that mandates grounded, citation-bearing answers and an explicit fallback string when context is insufficient.
8. Response returns the generated answer, a list of `Citation` objects, and an `ExecutionMetrics` block.

**API Contract — `POST /api/v1/query`**
- **Request Body:**
```json
{ "prompt": "string, min length 3", "top_k": "int, 1-10, default 4" }
```
- **Success Response (200):**
```json
{
  "answer": "string",
  "citations": [
    {
      "file_name": "string",
      "page_number": "int | null",
      "row_number": "int | null",
      "chunk_id": "string",
      "text_snippet": "string",
      "relevance_score": 0.0,
      "raw_logit": 0.0,
      "distance": 0.0
    }
  ],
  "metrics": {
    "retrieval_latency_ms": 0.0,
    "retrieval_breakdown": {
      "bm25_ms": 0.0, "dense_ms": 0.0, "chroma_io_ms": 0.0,
      "fusion_ms": 0.0, "rerank_ms": 0.0, "total_retrieval_ms": 0.0
    },
    "llm_latency_ms": 0.0,
    "total_latency_ms": 0.0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "estimated_cost_usd": 0.0
  }
}
```
- **Error Responses:** `400` (empty prompt, legacy route only), `500` (retrieval or generation failure).
- **Degraded-mode behavior:** If no `GROQ_API_KEY` is configured, the endpoint still performs full retrieval and returns citations, but `answer` is set to a static message indicating the missing API key rather than raising an error.

### 2.3 Workflow: List Indexed Documents
- **`GET /api/v1/documents`** returns an array of per-document summaries aggregated from all chunk metadata currently in ChromaDB: `doc_id`, `file_name`, `total_pages`, `chunk_count`. No pagination is implemented — the full corpus is always returned.

### 2.4 Workflow: Delete Document
- **`DELETE /api/v1/documents/{doc_id}`** deletes all ChromaDB records matching `where={"doc_id": doc_id}` and triggers a full BM25 index rebuild from the remaining corpus. Returns `404` if no matching records exist, otherwise `200` with a confirmation payload.

---

## 3. Assumptions & System Limits

These are explicit constants and implicit behaviors observed in the implementation, not aspirational targets:

- **Maximum upload size:** 10 MB (`MAX_FILE_SIZE_MB`), configurable via `.env`.
- **Chunking parameters:** `CHUNK_SIZE=1000` characters, `CHUNK_OVERLAP=200` characters, applied only to PDF/TXT/MD content (CSV is chunked by row, not by character count).
- **Top-K bounds:** query-time `top_k` is clamped to the range 1–10 at the schema level (`Field(ge=1, le=10)`); default is 4.
- **Hybrid retrieval pool size:** dense and sparse candidate pools are each capped at 10 (`DENSE_TOP_K`, `SPARSE_TOP_K`) before fusion, regardless of the requested `top_k`.
- **Single-collection architecture:** all documents across all formats share one ChromaDB collection (`document_intelligence`); there is no per-user or per-tenant isolation.
- **No authentication/authorization layer.** CORS is fully open (`allow_origins=["*"]`), and no API key or session mechanism gates access to any endpoint.
- **No incremental BM25 updates.** The sparse index is fully rebuilt from the entire corpus on every add and delete operation (`_rebuild_bm25_index`), which is O(n) in corpus size per mutation.
- **PDF text extraction is extraction-only.** Scanned/image-only PDFs with no embedded text layer will fail ingestion (`HTTP 400`) since no OCR fallback is implemented.
- **Embedding model selection is environment-driven.** If `OPENAI_API_KEY` is set, `text-embedding-3-small` (OpenAI) is used; otherwise the system falls back to the local `all-MiniLM-L6-v2` SentenceTransformer model. This is decided once at `VectorStoreService` initialization, not per-request.
- **LLM availability is optional.** The RAG service initializes without a functioning LLM if `GROQ_API_KEY` is absent, degrading query responses to "retrieval-only" mode (see 2.2).
- **Persistence:** ChromaDB is configured as a `PersistentClient` writing to `CHROMA_DB_DIR` (default `./chroma_db`) — data survives process restarts but is local-disk-bound (no distributed store).
- **Duplicate detection is content-based, not filename-based.** Uploading the same filename with different bytes creates a second, independent document; uploading identical bytes under any filename replaces the prior matching document.
- **Distance metric:** ChromaDB collection is explicitly configured with `hnsw:space: cosine`.