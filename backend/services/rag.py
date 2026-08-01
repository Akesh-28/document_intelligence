import os
import time
import threading
import ftfy
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from backend.config import settings
from backend.services.vector_store import vector_store_service
from backend.services.reranker import reranker_service
from backend.models.schemas import LatencyBreakdown

COST_PER_1M_PROMPT = 0.59
COST_PER_1M_COMPLETION = 0.79

SYSTEM_PROMPT = """You are an accurate, grounded Document Intelligence Assistant. 
Answer the user's question strictly using the provided context chunks.

Guidelines:
1. If the context does not contain sufficient information to answer the question, respond with:
   "No relevant context found in documents."
2. Do not invent, extrapolate, or use outside facts.
3. Include inline citations corresponding to the context sources used:
   - For document files (PDF, TXT, MD), cite as: [File: filename.ext, Page: X]
   - For CSV files, cite as: [File: filename.csv, Row: Y]

Context Chunks:
{context}
"""


def clean_text(text: str) -> str:
    """Fixes encoding artifacts (Mojibake) cleanly using ftfy with standard fallbacks."""
    if not text:
        return ""
    fixed = ftfy.fix_text(text)
    return (
        fixed.replace("â\x80\x93", "–")
        .replace("â\x80\x94", "—")
        .replace("â\x80\x99", "'")
        .replace("â\x80\x9c", '"')
        .replace("â\x80\x9d", '"')
    )


class RAGService:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", None)
        if not api_key or not str(api_key).strip():
            self.llm = None
        else:
            self.llm = ChatGroq(
                model=settings.LLM_MODEL,
                temperature=0.0,
                api_key=api_key  # type: ignore
            )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "{question}")
        ])

    def generate_rag_response(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        total_start = time.perf_counter()

        # --- RETRIEVAL PHASE ---
        retrieval_start = time.perf_counter()

        bm25_ms = 0.0
        fusion_ms = 0.0
        rerank_ms = 0.0

        # Dense Vector Retrieval + ChromaDB I/O Latency
        dense_candidates, dense_ms, chroma_io_ms = vector_store_service.search_similar_chunks(
            query=query,
            k=settings.DENSE_TOP_K if settings.ENABLE_HYBRID_SEARCH else top_k
        )

        # Sparse BM25 Retrieval Latency
        if settings.ENABLE_HYBRID_SEARCH:
            sparse_candidates, bm25_ms = vector_store_service.search_bm25(
                query=query,
                top_k=settings.SPARSE_TOP_K
            )

            # Reciprocal Rank Fusion Latency
            t_fusion_start = time.perf_counter()
            fused_candidates = reranker_service.reciprocal_rank_fusion(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates
            )
            fusion_ms = round((time.perf_counter() - t_fusion_start) * 1000.0, 2)
        else:
            fused_candidates = dense_candidates

        # Cross-Encoder Reranking Latency
        if settings.ENABLE_RERANKING and fused_candidates:
            t_rerank_start = time.perf_counter()
            final_chunks = reranker_service.rerank_chunks(
                query=query,
                candidates=fused_candidates,
                top_k=top_k
            )
            rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000.0, 2)
        else:
            final_chunks = fused_candidates[:top_k]

        retrieval_end = time.perf_counter()
        retrieval_ms = round((retrieval_end - retrieval_start) * 1000.0, 2)

        retrieval_breakdown = LatencyBreakdown(
            bm25_ms=bm25_ms,
            dense_ms=dense_ms,
            chroma_io_ms=chroma_io_ms,
            fusion_ms=fusion_ms,
            rerank_ms=rerank_ms,
            total_retrieval_ms=retrieval_ms
        )

        breakdown_dict = (
            retrieval_breakdown.model_dump()
            if hasattr(retrieval_breakdown, "model_dump")
            else retrieval_breakdown.dict()
        )

        if not final_chunks:
            total_end = time.perf_counter()
            return {
                "answer": "No relevant context found in documents.",
                "citations": [],
                "metrics": {
                    "retrieval_latency_ms": retrieval_ms,
                    "retrieval_breakdown": breakdown_dict,
                    "llm_latency_ms": 0.0,
                    "total_latency_ms": round((total_end - total_start) * 1000.0, 2),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "estimated_cost_usd": 0.0
                }
            }

        # --- CONTEXT ASSEMBLY ---
        context_blocks = []
        citations = []

        for idx, item in enumerate(final_chunks):
            meta = item.get("metadata", {})
            file_name = meta.get("file_name", "Unknown")
            page_num = meta.get("page_number")
            row_num = meta.get("row_number")

            chunk_id = item.get("chunk_id") or meta.get("chunk_id") or f"chunk_{idx}"
            raw_snippet = item.get("text", "")
            clean_snippet = clean_text(raw_snippet)

            loc_label = f"Row: {row_num}" if row_num is not None else f"Page: {page_num if page_num else 1}"
            context_blocks.append(f"[File: {file_name}, {loc_label}]\n{clean_snippet}")

            raw_logit = item.get("raw_logit")
            relevance_score = item.get("relevance_score", item.get("rrf_score", 0.0))

            citations.append({
                "file_name": file_name,
                "page_number": page_num,
                "row_number": row_num,
                "chunk_id": str(chunk_id),
                "text_snippet": clean_snippet,
                "distance": float(item.get("distance", 0.0)),
                "relevance_score": float(relevance_score),
                "raw_logit": raw_logit
            })

        formatted_context = "\n\n---\n\n".join(context_blocks)

        if not self.llm:
            total_end = time.perf_counter()
            return {
                "answer": "GROQ API key missing. Context retrieved successfully.",
                "citations": citations,
                "metrics": {
                    "retrieval_latency_ms": retrieval_ms,
                    "retrieval_breakdown": breakdown_dict,
                    "llm_latency_ms": 0.0,
                    "total_latency_ms": round((total_end - total_start) * 1000.0, 2),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "estimated_cost_usd": 0.0
                }
            }

        # --- LLM GENERATION PHASE ---
        llm_start = time.perf_counter()

        prompt_tokens = 0
        completion_tokens = 0

        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "context": formatted_context,
                "question": query
            })
            answer_text = str(response.content).strip()

            # Extraction of token counts across standard attribute structures
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_meta = response.usage_metadata
                prompt_tokens = usage_meta.get("input_tokens", usage_meta.get("prompt_tokens", 0))
                completion_tokens = usage_meta.get("output_tokens", usage_meta.get("completion_tokens", 0))

            if prompt_tokens == 0 and hasattr(response, "response_metadata"):
                token_usage = response.response_metadata.get("token_usage", {})
                prompt_tokens = token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
                completion_tokens = token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))

        except Exception as e:
            answer_text = f"Error generating response from LLM: {str(e)}"

        llm_end = time.perf_counter()
        llm_ms = round((llm_end - llm_start) * 1000.0, 2)
        total_end = time.perf_counter()
        total_ms = round((total_end - total_start) * 1000.0, 2)

        cost_usd = (
            (prompt_tokens / 1_000_000) * COST_PER_1M_PROMPT +
            (completion_tokens / 1_000_000) * COST_PER_1M_COMPLETION
        )

        return {
            "answer": answer_text,
            "citations": citations,
            "metrics": {
                "retrieval_latency_ms": retrieval_ms,
                "retrieval_breakdown": breakdown_dict,
                "llm_latency_ms": llm_ms,
                "total_latency_ms": total_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_cost_usd": round(cost_usd, 6)
            }
        }


# --- Lazy Singleton Pattern ---
_rag_service_instance = None
_rag_service_lock = threading.Lock()


def get_rag_service() -> RAGService:
    """Returns a singleton instance of RAGService, initialized on demand."""
    global _rag_service_instance
    if _rag_service_instance is None:
        with _rag_service_lock:
            if _rag_service_instance is None:
                _rag_service_instance = RAGService()
    return _rag_service_instance