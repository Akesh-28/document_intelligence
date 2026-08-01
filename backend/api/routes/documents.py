from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from backend.models.schemas import (
    UploadResponse,
    DocumentMetadata,
    ErrorResponse
)
from backend.services.ingestion import process_and_chunk_document
from backend.services.vector_store import vector_store_service

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {"pdf", "txt", "md", "csv"}


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and index a document"
)
async def upload_document(file: UploadFile = File(...)):
    """
    Parses document bytes, validates extensions, chunks text/rows with metadata,
    and indexes embeddings in ChromaDB.
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
        # Process, parse, and chunk document
        doc_id, chunks, metadatas, total_pages = process_and_chunk_document(
            file_bytes=file_bytes,
            filename=file.filename
        )

        # Add to ChromaDB vector store
        vector_store_service.add_documents(chunks=chunks, metadatas=metadatas)

        doc_meta = DocumentMetadata(
            doc_id=doc_id,
            file_name=file.filename,
            file_type=f".{ext}",
            page_count=total_pages,
            total_chunks=len(chunks)
        )

        return UploadResponse(
            message="Document indexed successfully.",
            document=doc_meta
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index document: {str(e)}"
        )


@router.get(
    "",
    summary="List all indexed documents"
)
async def list_documents():
    """
    Returns unique list of all documents stored in ChromaDB vector store.
    """
    try:
        documents = vector_store_service.list_indexed_documents()
        return documents
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete(
    "/{doc_id}",
    summary="Delete document vectors by doc_id"
)
async def delete_document(doc_id: str):
    """
    Purges all chunks and metadata corresponding to doc_id from ChromaDB.
    """
    deleted = vector_store_service.delete_document_by_id(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found in index."
        )
    return {"message": "Document successfully removed from index.", "doc_id": doc_id}