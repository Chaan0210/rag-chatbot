import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_name: str = os.getenv("PROJECT_NAME", "RAG Chatbot")
    api_prefix: str = os.getenv("API_PREFIX", "/api")

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-5")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    reasoning_effort: str = os.getenv("OPENAI_REASONING_EFFORT", "minimal")

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_db",
    )

    cors_allow_origins: str = os.getenv("CORS_ALLOW_ORIGINS", "*")
    data_dir: str = os.getenv("DATA_DIR", "data")
    auto_ingest_on_startup: bool = os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    embedding_max_chars: int = int(os.getenv("EMBEDDING_MAX_CHARS", "8000"))

    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "50"))
    keyword_top_k: int = int(os.getenv("KEYWORD_TOP_K", "50"))
    keyword_candidate_pool_size: int = int(os.getenv("KEYWORD_CANDIDATE_POOL_SIZE", "2000"))
    keyword_recent_docs_limit: int = int(os.getenv("KEYWORD_RECENT_DOCS_LIMIT", "8"))
    enable_trigram_search: bool = os.getenv("ENABLE_TRIGRAM_SEARCH", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    fused_top_k: int = int(os.getenv("FUSED_TOP_K", "50"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "8"))
    enable_reranker: bool = os.getenv("ENABLE_RERANKER", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
    history_summary_turns: int = int(os.getenv("HISTORY_SUMMARY_TURNS", "10"))
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "32000"))
    max_quote_chars: int = int(os.getenv("MAX_QUOTE_CHARS", "280"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "180"))

    multi_query_count: int = int(os.getenv("MULTI_QUERY_COUNT", "5"))
    retrieval_max_attempts: int = int(os.getenv("RETRIEVAL_MAX_ATTEMPTS", "3"))
    retrieval_quality_threshold: float = float(os.getenv("RETRIEVAL_QUALITY_THRESHOLD", "0.55"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
