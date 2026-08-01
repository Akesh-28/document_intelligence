from fastapi import APIRouter, HTTPException, status
from backend.models.schemas import QueryRequest, QueryResponse, ErrorResponse
from backend.services.rag import rag_service

router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Query the RAG engine with real-time configuration"
)
async def query_rag(request: QueryRequest):
    """
    Executes dense/sparse hybrid search, cross-encoder re-ranking, and LLM answer generation with telemetry.
    """
    try:
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