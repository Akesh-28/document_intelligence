import os
import hashlib
from typing import List

# Enforce thread restrictions BEFORE importing heavy ML frameworks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.services.ingestion import process_and_chunk_document
from backend.services.vector_store import vector_store_service
from backend.services.rag import get_rag_service

from backend.models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentMetadata,
    UploadResponse as DocumentUploadResponse,
)

app = FastAPI(
    title="Intelligent Document Understanding & Retrieval Engine API",
    version="1.0.0",
    description="Production-ready FastAPI backend for PDF/TXT/MD/CSV ingestion, ChromaDB vector indexing, and RAG retrieval."
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

@app.get("/health", status_code=status.HTTP_200_OK, summary="Health Check")
async def health_check():
    """Liveness and readiness probe for Docker / Render container health checks."""
    return {"status": "ok"}


@app.post(
    "/api/v1/documents/upload", 
    response_model=DocumentUploadResponse, 
    status_code=status.HTTP_200_OK,
    summary="Upload and index a document (.pdf, .txt, .md, .csv)"
)
async def upload_document(file: UploadFile = File(...)):
    """
    Parses document bytes, enforces SHA-256 duplicate detection, chunking with metadata,
    and indexes embeddings in ChromaDB. Replaces existing document if identical hash exists.
    """
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
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        existing_doc_id = vector_store_service.find_document_by_hash(content_hash)
        
        if existing_doc_id:
            vector_store_service.delete_document_by_id(existing_doc_id)

        doc_id, chunks, metadatas, total_pages = process_and_chunk_document(
            file_bytes=file_bytes,
            filename=file.filename
        )
        
        vector_store_service.add_documents(chunks=chunks, metadatas=metadatas)
        
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
        rag_service = get_rag_service()
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
    documents = vector_store_service.list_indexed_documents()
    return [DocumentInfoSchema(**doc) for doc in documents]


@app.delete(
    "/api/v1/documents/{doc_id}", 
    response_model=DeleteResponse,
    summary="Delete document vectors by doc_id"
)
async def delete_document(doc_id: str):
    deleted = vector_store_service.delete_document_by_id(doc_id)
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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)