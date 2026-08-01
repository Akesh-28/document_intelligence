from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


# --- Document Metadata Schemas ---
class DocumentMetadata(BaseModel):
    doc_id: str = Field(..., description="Unique UUID assigned to document")
    file_name: str = Field(..., description="Original filename uploaded")
    file_type: str = Field(..., description="Extension (.pdf, .txt, .md, .csv)")
    page_count: int = Field(default=1, description="Total pages or structural units")
    total_chunks: int = Field(..., description="Number of vector chunks indexed")
    upload_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO UTC Timestamp",
    )


class UploadResponse(BaseModel):
    message: str
    document: DocumentMetadata
    replaced_doc_id: Optional[str] = Field(
        None, description="ID of replaced document if duplicate hash existed"
    )


# --- Citation & RAG Query Schemas ---
class Citation(BaseModel):
    file_name: str
    page_number: Optional[int] = None
    row_number: Optional[int] = None
    chunk_id: str
    text_snippet: str
    relevance_score: float = Field(
        default=0.0,
        description="Confidence score or normalized relevance (0.0 - 1.0 or percentage)",
    )
    raw_logit: Optional[float] = Field(
        None,
        description="Internal cross-encoder logit before sigmoid normalization for debugging.",
    )
    distance: Optional[float] = Field(
        None,
        description="Raw vector distance metric if available",
    )


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="User query question")
    top_k: Optional[int] = Field(default=4, ge=1, le=10, description="Top K vectors to fetch")


class LatencyBreakdown(BaseModel):
    bm25_ms: float = Field(default=0.0, description="BM25 keyword search execution time (ms)")
    dense_ms: float = Field(default=0.0, description="Dense embedding model encoding execution time (ms)")
    chroma_io_ms: float = Field(default=0.0, description="ChromaDB vector persistence lookup time (ms)")
    fusion_ms: float = Field(default=0.0, description="Reciprocal Rank Fusion execution time (ms)")
    rerank_ms: float = Field(default=0.0, description="Cross-encoder reranking execution time (ms)")
    total_retrieval_ms: float = Field(default=0.0, description="Aggregate retrieval phase execution time (ms)")


class ExecutionMetrics(BaseModel):
    retrieval_latency_ms: float
    retrieval_breakdown: LatencyBreakdown
    llm_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    metrics: ExecutionMetrics


# --- Error Response Schema ---
class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[str] = None