"""
philorag.config
───────────────
Configuração centralizada via Pydantic Settings.
Carrega variáveis de .env com fallback para valores saudáveis.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


class EmbeddingProvider(str, Enum):
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openrouter_api_key: str = ""

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OLLAMA
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768

    # ── Vector store ──────────────────────────────────────────────────────────
    vector_store_path: Path = Path("./data/chromadb")
    chroma_collection_name: str = "philorag"

    # ── Document storage ──────────────────────────────────────────────────────
    documents_path: Path = Path("./data/documents")
    processed_path: Path = Path("./data/processed")
    database_url: str = "sqlite+aiosqlite:///./data/philorag.db"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    hierarchical_levels: int = 4

    # ── RAG ───────────────────────────────────────────────────────────────────
    rag_top_k: int = 8
    rag_rerank: bool = True

    # ── App ───────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Cria diretórios necessários se não existirem."""
        for p in (self.vector_store_path, self.documents_path, self.processed_path):
            p.mkdir(parents=True, exist_ok=True)
        # diretório do SQLite
        db_path = self.database_url.replace("sqlite+aiosqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
