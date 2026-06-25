# app/database/db.py
from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON, Index
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column

from app.config import settings

# ── Engine e sessão assíncrona ──────────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


# ── Modelos ORM ─────────────────────────────────────────────────────────────────

class DocumentORM(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    filepath: Mapped[str] = mapped_column(String(512), nullable=False)
    titulo: Mapped[Optional[str]] = mapped_column(String(255))
    autor: Mapped[Optional[str]] = mapped_column(String(255))
    obra: Mapped[Optional[str]] = mapped_column(String(255))
    ano: Mapped[Optional[int]] = mapped_column(Integer)
    idioma: Mapped[Optional[str]] = mapped_column(String(10), default="pt")
    status: Mapped[str] = mapped_column(String(20), default="indexed")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    summary_work: Mapped[Optional[str]] = mapped_column(Text)
    summary_chapters: Mapped[Optional[str]] = mapped_column(Text)
    concept_map: Mapped[Optional[str]] = mapped_column(Text)

    palavras_chave: Mapped[Optional[str]] = mapped_column(Text)
    conceitos: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chunks: Mapped[List["ChunkORM"]] = relationship(back_populates="document", cascade="all, delete-orphan")

    def get_palavras_chave(self) -> list:
        return json.loads(self.palavras_chave) if self.palavras_chave else []

    def get_conceitos(self) -> list:
        return json.loads(self.conceitos) if self.conceitos else []


class ChunkORM(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_level", "level"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    parent_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    obra: Mapped[Optional[str]] = mapped_column(String(255))
    capitulo: Mapped[Optional[str]] = mapped_column(String(255))
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    vector_id: Mapped[Optional[str]] = mapped_column(String(100))

    document: Mapped["DocumentORM"] = relationship(back_populates="chunks")
    children: Mapped[List["ChunkORM"]] = relationship(
        "ChunkORM",
        remote_side=[id],
        backref="parent",
        cascade="all, delete-orphan"
    )


# ── Inicialização e dependência ─────────────────────────────────────────────────

async def init_db() -> None:
    """Cria as tabelas no banco de dados (executado no startup)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Dependência para obter uma sessão assíncrona."""
    async with AsyncSessionLocal() as session:
        yield session
