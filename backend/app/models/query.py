"""
philorag.models.query
─────────────────────
Modelos para consultas RAG e respostas anotadas.
"""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class QueryMode(str, Enum):
    """
    Modos de consulta ao corpus filosófico.
    
    EXEGETICAL  → reconstrução conceitual / leitura imanente
    BIBLIOGRAPHIC → rastreamento de comentadores
    COMPARATIVE → comparação entre intérpretes
    DOSSIER     → dossiê temático completo
    FREE        → consulta livre
    """
    EXEGETICAL    = "exegetical"
    BIBLIOGRAPHIC = "bibliographic"
    COMPARATIVE   = "comparative"
    DOSSIER       = "dossier"
    FREE          = "free"


class QueryRequest(BaseModel):
    """Requisição de consulta ao pipeline RAG."""
    question: str
    mode: QueryMode = QueryMode.FREE

    # Filtros de corpus
    document_ids: list[UUID] = Field(default_factory=list)  # vazio = corpus completo
    obra_sigla: str = ""           # ex.: "ZA", "ABM"
    autor: str = ""
    ano_min: int | None = None
    ano_max: int | None = None

    # Controle de contexto
    top_k: int = 8
    include_summaries: bool = True
    include_concept_map: bool = True
    include_adjacent_chunks: bool = True   # chunks vizinhos para contexto contínuo

    # LLM
    llm_model: str = ""            # override do modelo default
    temperature: float = 0.3       # baixo para análise filosófica precisa
    max_tokens: int = 2048


class ChunkCitation(BaseModel):
    """Chunk recuperado com pontuação de relevância."""
    chunk_id: UUID
    document_id: UUID
    score: float
    level: int
    title: str
    text: str
    metadata_obra: str
    metadata_autor: str
    metadata_ano: str
    page_ref: str = ""


class RAGResponse(BaseModel):
    """Resposta completa do pipeline RAG com citações."""
    question: str
    mode: QueryMode
    answer: str

    # Fontes utilizadas (ordenadas por relevância)
    citations: list[ChunkCitation] = Field(default_factory=list)

    # Contexto hierárquico injetado (para transparência)
    work_summaries_used: list[str] = Field(default_factory=list)
    chapter_summaries_used: list[str] = Field(default_factory=list)
    concept_map_used: list[str] = Field(default_factory=list)

    # Métricas
    chunks_retrieved: int = 0
    tokens_in_context: int = 0
    model_used: str = ""
    latency_ms: int = 0


class DossierRequest(BaseModel):
    """Requisição de dossiê temático."""
    theme: str                          # ex.: "animalidade", "corpo", "além-do-homem"
    include_primary: bool = True        # textos primários
    include_commentators: bool = True   # comentadores
    include_citations: bool = True
    export_format: str = "markdown"     # markdown | docx | pdf
    citation_style: str = "abnt"        # abnt | chicago | apa


class DossierSection(BaseModel):
    title: str
    content: str
    sources: list[ChunkCitation] = Field(default_factory=list)


class Dossier(BaseModel):
    theme: str
    sections: list[DossierSection] = Field(default_factory=list)
    bibliography: list[str] = Field(default_factory=list)
    concept_map: dict[str, list[str]] = Field(default_factory=dict)
    export_path: str = ""
