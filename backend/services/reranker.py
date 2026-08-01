import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from backend.config import settings
import math
logger = logging.getLogger(__name__)
def logit_to_confidence_percentage(logit_value: float) -> float:
    """
    Converts a raw cross-encoder logit into a 0-100 confidence percentage
    using a sigmoid transform, with overflow protection for extreme logits.
    """
    if logit_value >= 20:
        return 100.0
    if logit_value <= -20:
        return 0.0
    sigmoid = 1.0 / (1.0 + math.exp(-logit_value))
    return round(sigmoid * 100, 2)
class RerankerService:
    def __init__(self):
        self.model = None
        if settings.ENABLE_RERANKING:
            try:
                logger.info(f"Loading Cross-Encoder model: {settings.RERANKER_MODEL}")
                self.model = CrossEncoder(settings.RERANKER_MODEL)
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder model '{settings.RERANKER_MODEL}': {e}")
                self.model = None
        
    def reciprocal_rank_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combines Dense and Sparse search results using Reciprocal Rank Fusion (RRF).
        RRF Score = 1.0 / (k + rank)
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Process Dense results
        for rank, item in enumerate(dense_results, start=1):
            chunk_id = item["metadata"].get("chunk_id", item["text"])
            chunk_map[chunk_id] = item
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

        # Process Sparse (BM25) results
        for rank, item in enumerate(sparse_results, start=1):
            chunk_id = item["metadata"].get("chunk_id", item["text"])
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = item
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

        # Sort candidates by combined RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        fused_results = []
        for cid in sorted_chunk_ids:
            chunk_item = chunk_map[cid].copy()
            chunk_item["rrf_score"] = rrf_scores[cid]
            fused_results.append(chunk_item)

        return fused_results

    def rerank_chunks(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Scores query-candidate pairs using the Cross-Encoder model and returns Top-K.
        Applies Min-Max normalization to standardise relative confidence (0.0 to 1.0).
        """
        if not candidates:
            return []

        if not self.model:
            # Fallback to candidate list directly if cross-encoder isn't loaded
            for item in candidates:
                item["relevance_score"] = 0.5
                item["raw_logit"] = 0.0
            return candidates[:top_k]

        # Prepare sentence pairs: (Query, Chunk Text)
        pairs = [(query, c["text"]) for c in candidates]
        raw_logits = self.model.predict(pairs)

        # Min-Max Scaling setup
        min_logit = float(min(raw_logits))
        max_logit = float(max(raw_logits))
        logit_range = max_logit - min_logit

        reranked_results = []
        for idx, item in enumerate(candidates):
            raw_logit = float(raw_logits[idx])
            
            if logit_range > 0:
                normalized_score = round((raw_logit - min_logit) / (logit_range + 1e-8), 4)
            else:
                normalized_score = 1.0

            item_copy = dict(item)
            item_copy["raw_logit"] = raw_logit
            item_copy["relevance_score"] = normalized_score
            reranked_results.append(item_copy)

        # Sort by raw logit descending
        reranked_results.sort(key=lambda x: x["raw_logit"], reverse=True)
        return reranked_results[:top_k]


# Global singleton instance
reranker_service = RerankerService()