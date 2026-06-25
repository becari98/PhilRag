"""
philorag.models.document
────────────────────────
Modelos de domínio: Document, Chunk, Metadata.
Representam a hierarquia: Obra → Capítulo → Parágrafo → Sentença.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class ChunkLevel(int, Enum):
    """Nível hierárquico do chunk."""
    WORK      = 1   # Obra completa
    CHAPTER   = 2   # Capítulo / discurso / parte
    PARAGRAPH = 3   # Parágrafo
    SENTENCE  = 4   # Sentença (opcional)


class DocumentStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    INDEXED    = "indexed"
    ERROR      = "error"


class ChunkStrategy(str, Enum):
    HIERARCHICAL = "hierarchical"   # por estrutura do documento
    TOKEN        = "token"          # por número de tokens
    PARAGRAPH    = "paragraph"      # por parágrafos
    APHORISM     = "aphorism"       # por aforismos (Nietzsche, Wittgenstein…)
    MANUAL       = "manual"         # delimitadores manuais


# ── Metadata ──────────────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    """Metadados extraídos de um documento."""
    titulo: str = ""
    autor: str = ""
    ano: str = ""
    idioma: str = "pt"
    obra: str = ""                   # sigla canônica (ex.: ZA, ABM, EH)
    capitulo: str = ""
    secao: str = ""
    palavras_chave: list[str] = Field(default_factory=list)
    conceitos: list[str] = Field(default_factory=list)
    citacoes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class ConceptRelation(BaseModel):
    """Relação conceitual extraída do mapa semântico."""
    conceito: str
    relacionado_com: list[str] = Field(default_factory=list)
    obras_de_referencia: list[str] = Field(default_factory=list)
    comentadores: list[str] = Field(default_factory=list)


# ── Core models ───────────────────────────────────────────────────────────────

class Document(BaseModel):
    """Representa um documento (obra) na biblioteca."""
    id: UUID = Field(default_factory=uuid4)
    filename: str
    filepath: str
    mimetype: str = ""
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    # Sumários hierárquicos (gerados após indexação)
    summary_work: str = ""       # sumário da obra completa
    summary_chapters: dict[str, str] = Field(default_factory=dict)  # {titulo_cap: sumário}

    # Mapa conceitual
    concept_map: list[ConceptRelation] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    chunk_count: int = 0


class Chunk(BaseModel):
    """
    Unidade atômica de texto indexada no vector store.
    
    Hierarquia:
        level=1 → obra completa (summary somente; não indexado como vetor)
        level=2 → capítulo / discurso
        level=3 → parágrafo (chunk primário de busca)
        level=4 → sentença (opcional, granularidade fina)
    """
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    parent_id: UUID | None = None    # referência ao chunk de nível superior

    level: ChunkLevel = ChunkLevel.PARAGRAPH
    position: int = 0               # ordem dentro do pai
    title: str = ""                 # título do capítulo/seção (se disponível)

    # Conteúdo
    text: str
    markdown: str = ""              # versão formatada
    summary: str = ""               # sumário gerado por LLM

    # Metadados herdados do documento + específicos
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    # Contexto hierárquico (preenchido na injeção de contexto para o LLM)
    context_above: str = ""         # texto do chunk pai (level-1)
    context_below: str = ""         # sumário do nível superior

    # Vector store reference
    embedding_id: str = ""

    token_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Request / Response schemas ────────────────────────────────────────────────

class DocumentCreate(BaseModel):
    """Schema para upload de documento."""
    obra: str = ""
    autor: str = ""
    ano: str = ""
    idioma: str = "pt"
    chunk_strategy: ChunkStrategy = ChunkStrategy.HIERARCHICAL
    chunk_size: int = 512
    chunk_overlap: int = 64
    generate_summaries: bool = True
    generate_concept_map: bool = True


class DocumentRead(BaseModel):
    """Schema de leitura pública."""
    id: UUID
    filename: str
    status: DocumentStatus
    metadata: DocumentMetadata
    chunk_count: int
    created_at: datetime


class ChunkRead(BaseModel):
    """Chunk serializado para resposta de API."""
    id: UUID
    document_id: UUID
    level: ChunkLevel
    title: str
    text: str
    summary: str
    metadata: DocumentMetadata
    position: int
    token_count: int
