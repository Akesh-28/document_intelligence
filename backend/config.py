import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Application Config
    APP_NAME: str = "Intelligent Document Understanding Engine"
    API_V1_STR: str = "/api/v1"
    
    # Storage & File Constraints
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md", ".csv"}
    CHROMA_DB_DIR: str = str(BASE_DIR / "chroma_db")
    
    # RAG / Embedding Hyperparameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    DEFAULT_TOP_K: int = 4
    DISTANCE_METRIC: str = "cosine"
    
    # Hybrid Search & Re-ranking Config (ADD THESE LINES)
    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_RERANKING: bool = True
    DENSE_TOP_K: int = 10
    SPARSE_TOP_K: int = 10
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Groq API & Models
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.3-70b-versatile" 
    
    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()