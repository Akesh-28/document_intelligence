import os
import hashlib
from typing import List, Optional
from contextlib import asynccontextmanager

# Enforce thread restrictions BEFORE importing heavy ML frameworks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.services.ingestion import process_and_chunk_document
from backend.models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentMetadata,
    UploadResponse as DocumentUploadResponse,
)

# --- Lazy Service Singletons ---
_vector_store_service = None
_rag_service = None


def get_vector_store():
    """Lazy loader for Vector Store Service to prevent startup delays."""
    global _vector_store_service
    if _vector_store_service is None:
        from backend.services.vector_store import vector_store_service
        _vector_store_service = vector_store_service
    return _vector_store_service


def get_rag():
    """Lazy loader for RAG Service to prevent startup delays."""
    global _rag_service
    if _rag_service is None:
        from backend.services.rag import get_rag_service
        _rag_service = get_rag_service()
    return _rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan: Ensures server port opens immediately before heavy models warm up."""
    # Preheat services after app startup
    try:
        get_vector_store()
        get_rag()
    except Exception as e:
        print(f"Warning during service warmup: {e}")
    yield


app = FastAPI(
    title="Intelligent Document Understanding & Retrieval Engine API",
    version="1.0.0",
    description="Production-ready FastAPI backend for PDF/TXT/MD/CSV ingestion, ChromaDB vector indexing, and RAG retrieval.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {"pdf", "txt", "md", "csv"}


# --- API Specific Schemas ---

class DocumentInfoSchema(BaseModel):
    doc_id: str
    file_name: str
    total_pages: int
    chunk_count: int


class DeleteResponse(BaseModel):
    message: str
    doc_id: str


# --- API Endpoints ---

@app.get("/", status_code=status.HTTP_200_OK, summary="Root Health Check")
@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check")
@app.get("/api/v1/health", status_code=status.HTTP_200_OK, summary="API Health Check")
async def health_check():
    """Liveness probe for Render container health checks."""
    return {"status": "ok"}


@app.post(
    "/api/v1/documents/upload", 
    response_model=DocumentUploadResponse, 
    status_code=status.HTTP_200_OK,
    summary="Upload and index a document (.pdf, .txt, .md, .csv)"
)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: .{ext}. Allowed extensions are: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    file_bytes = await file.read()
    
    try:
        vector_store = get_vector_store()
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        existing_doc_id = vector_store.find_document_by_hash(content_hash)
        
        if existing_doc_id:
            vector_store.delete_document_by_id(existing_doc_id)

        doc_id, chunks, metadatas, total_pages = process_and_chunk_document(
            file_bytes=file_bytes,
            filename=file.filename
        )
        
        vector_store.add_documents(chunks=chunks, metadatas=metadatas)
        
        doc_metadata = DocumentMetadata(
            doc_id=doc_id,
            file_name=file.filename,
            file_type=f".{ext}",
            page_count=total_pages,
            total_chunks=len(chunks)
        )

        return DocumentUploadResponse(
            message="Successfully indexed document.",
            document=doc_metadata,
            replaced_doc_id=existing_doc_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {str(e)}"
        )


@app.post(
    "/api/v1/query", 
    response_model=QueryResponse,
    summary="Submit query to RAG retrieval pipeline"
)
async def query_documents(request: QueryRequest):
    if not request.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )
        
    try:
        rag_service = get_rag()
        result = rag_service.generate_rag_response(
            query=request.prompt, 
            top_k=request.top_k or 4
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get(
    "/api/v1/documents", 
    response_model=List[DocumentInfoSchema],
    summary="List all indexed documents"
)
async def list_documents():
    vector_store = get_vector_store()
    documents = vector_store.list_indexed_documents()
    return [DocumentInfoSchema(**doc) for doc in documents]


@app.delete(
    "/api/v1/documents/{doc_id}", 
    response_model=DeleteResponse,
    summary="Delete document vectors by doc_id"
)
async def delete_document(doc_id: str):
    vector_store = get_vector_store()
    deleted = vector_store.delete_document_by_id(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found in index."
        )
        
    return DeleteResponse(
        message="Document successfully removed from index.",
        doc_id=doc_id
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)