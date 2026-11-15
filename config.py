import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEDIRAG_", arbitrary_types_allowed=True)
    # Base paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    INDICES_DIR: Path = DATA_DIR / "indices"
    EVAL_RESULTS_DIR: Path = DATA_DIR / "eval_results"

    # Embedding & Reranker models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DEVICE: str = "cpu"  # CPU for lightweight MiniLM ensures zero VRAM contention with Ollama

    # Chunking configuration
    CHUNK_SIZE: int = 500  # characters
    CHUNK_OVERLAP: int = 100

    # Retrieval parameters
    TOP_K_VECTOR: int = 10
    TOP_K_BM25: int = 10
    TOP_K_RRF: int = 10
    RRF_K: int = 60
    TOP_K_FINAL: int = 4  # context passages passed to LLM
    MIN_RELEVANCE_SCORE: float = -4.0  # reranker minimum threshold for out-of-scope check

    # Ollama LLM parameters
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_FALLBACK_MODEL: str = "qwen2.5-coder:3b"
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT: float = 60.0

    # Server settings
    BACKEND_HOST: str = "127.0.0.1"
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 8501

settings = Settings()

# Ensure directories exist
settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
settings.INDICES_DIR.mkdir(parents=True, exist_ok=True)
settings.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
