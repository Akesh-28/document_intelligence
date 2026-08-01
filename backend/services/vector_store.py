import os
import re
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "document_intelligence"

ChromaMetadata = Dict[str, Union[str, int, float, bool]]


def tokenize_text(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 indexing."""
    return re.findall(r"\w+", text.lower())


class VectorStoreService:
    def __init__(self, persist_dir: str = PERSIST_DIRECTORY):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key.strip():
            self.embedding_fn: Any = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small"
            )
        else:
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_documents: List[str] = []
        self.bm25_metadatas: List[Dict[str, Any]] = []
        
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        """Rebuilds the sparse BM25 index from all documents currently in ChromaDB."""
        all_data = self.collection.get(include=["documents", "metadatas"])
        
        docs = all_data.get("documents") if all_data else None
        metas = all_data.get("metadatas") if all_data else None

        if docs and isinstance(docs, list) and len(docs) > 0:
            self.bm25_documents = docs
            self.bm25_metadatas = metas if metas else [{}] * len(docs)
            corpus_tokens = [tokenize_text(doc) for doc in self.bm25_documents]
            self.bm25_index = BM25Okapi(corpus_tokens)
        else:
            self.bm25_index = None
            self.bm25_documents = []
            self.bm25_metadatas = []

    def add_documents(
        self, 
        chunks: List[str], 
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        ids = [str(m.get("chunk_id", f"chunk_{i}")) for i, m in enumerate(metadatas)]
        
        sanitized_metadatas: List[ChromaMetadata] = []
        for m in metadatas:
            clean_meta: ChromaMetadata = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            sanitized_metadatas.append(clean_meta)

        self.collection.add(
            documents=chunks,
            metadatas=sanitized_metadatas,  # type: ignore
            ids=ids
        )
        self._rebuild_bm25_index()
        return ids

    def search_similar_chunks(
        self, 
        query: str, 
        k: int = 4
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """
        Dense similarity search using ChromaDB.
        Returns (results, dense_encoding_latency_ms, chroma_io_latency_ms).
        """
        # 1. Dense Embedding Generation Latency
        t_embed_start = time.perf_counter()
        if hasattr(self.embedding_fn, "__call__"):
            _ = self.embedding_fn([query])
        dense_ms = (time.perf_counter() - t_embed_start) * 1000.0

        # 2. ChromaDB I/O Latency
        t_io_start = time.perf_counter()
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        chroma_io_ms = (time.perf_counter() - t_io_start) * 1000.0
        
        formatted_results = []
        if results:
            docs_list = results.get("documents")
            metas_list = results.get("metadatas")
            dists_list = results.get("distances")

            if docs_list and len(docs_list) > 0 and docs_list[0]:
                docs = docs_list[0]
                metas = metas_list[0] if metas_list else [{}] * len(docs)
                dists = dists_list[0] if dists_list else [0.0] * len(docs)
                
                for doc, meta, dist in zip(docs, metas, dists):
                    formatted_results.append({
                        "text": doc,
                        "metadata": meta or {},
                        "distance": float(dist)
                    })
                
        return formatted_results, round(dense_ms, 2), round(chroma_io_ms, 2)

    def search_bm25(self, query: str, top_k: int = 10) -> Tuple[List[Dict[str, Any]], float]:
        """
        Sparse keyword search using BM25Okapi.
        Returns (results, bm25_latency_ms).
        """
        t_start = time.perf_counter()
        if not self.bm25_index or not self.bm25_documents:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return [], round(elapsed_ms, 2)

        query_tokens = tokenize_text(query)
        scores = self.bm25_index.get_scores(query_tokens)
        
        scored_docs = list(zip(scores, self.bm25_documents, self.bm25_metadatas))
        scored_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)

        results = []
        for score, doc_text, meta in scored_docs[:top_k]:
            if score > 0:
                results.append({
                    "text": doc_text,
                    "metadata": meta,
                    "bm25_score": float(score)
                })
        
        bm25_ms = (time.perf_counter() - t_start) * 1000.0
        return results, round(bm25_ms, 2)

    def delete_document_by_id(self, doc_id: str) -> bool:
        existing = self.collection.get(where={"doc_id": doc_id})
        if existing and existing.get("ids"):
            self.collection.delete(where={"doc_id": doc_id})
            self._rebuild_bm25_index()
            return True
        return False

    def find_document_by_hash(self, content_hash: str) -> Optional[str]:
        if not content_hash or not str(content_hash).strip():
            return None

        target_hash = str(content_hash).strip()

        try:
            results = self.collection.get(
                where={"content_hash": target_hash},
                include=["metadatas"]
            )
            if results and results.get("metadatas"):
                for meta in results["metadatas"]:
                    if isinstance(meta, dict) and meta.get("doc_id"):
                        return str(meta["doc_id"])
        except Exception:
            pass

        for meta in self.bm25_metadatas:
            if isinstance(meta, dict):
                if str(meta.get("content_hash", "")).strip() == target_hash and meta.get("doc_id"):
                    return str(meta["doc_id"])

        try:
            all_records = self.collection.get(include=["metadatas"])
            if all_records and all_records.get("metadatas"):
                for meta in all_records["metadatas"]:
                    if isinstance(meta, dict) and str(meta.get("content_hash", "")).strip() == target_hash:
                        if meta.get("doc_id"):
                            return str(meta["doc_id"])
        except Exception:
            pass

        return None
    
    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        results = self.collection.get(include=["metadatas"])
        
        if not results or not results.get("metadatas"):
            return []

        metadatas = results["metadatas"]
        docs_summary: Dict[str, Dict[str, Any]] = {}
        
        for meta in metadatas:
            if not meta or not isinstance(meta, dict):
                continue
                
            doc_id = str(meta.get("doc_id", "default_doc"))
            file_name = str(meta.get("file_name", "Unknown File"))
            total_pages = meta.get("total_pages", 1)
            
            if doc_id not in docs_summary:
                docs_summary[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "total_pages": total_pages,
                    "chunk_count": 1
                }
            else:
                docs_summary[doc_id]["chunk_count"] += 1
                
        return list(docs_summary.values())


vector_store_service = VectorStoreService()