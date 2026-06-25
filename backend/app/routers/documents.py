"""
philorag.routers.documents
────────────────────────────
Endpoints REST para gerenciamento de documentos.

POST   /documents/         →  upload + indexação
GET    /documents/         →  lista da biblioteca
GET    /documents/{id}     →  detalhes de um documento
DELETE /documents/{id}     →  remove documento
GET    /documents/{id}/chunks  →  chunks hierárquicos
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.db import ChunkORM, DocumentORM, get_db
from app.models.document import ChunkStrategy, DocumentCreate, DocumentRead
from app.services.indexer import IndexingService

router = APIRouter(prefix="/documents", tags=["documents"])
indexer = IndexingService()


# ── Upload & indexação ─────────────────────────────────────────────────────────

@router.post("/", response_model=dict, status_code=202)
async def upload_document(
    file: UploadFile = File(description="Arquivo a indexar (PDF, DOCX, EPUB, TXT…)"),
    obra: str = Form(""),
    autor: str = Form(""),
    ano: str = Form(""),
    idioma: str = Form("pt"),
    chunk_strategy: ChunkStrategy = Form(ChunkStrategy.HIERARCHICAL),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(64),
    generate_summaries: bool = Form(True),
    generate_concept_map: bool = Form(True),
    db: AsyncSession = Depends(get_db),
):
    """
    Recebe um arquivo, salva no disco e inicia indexação.
    Retorna o document_id imediatamente; a indexação é síncrona (MVP).
    """
    settings.ensure_dirs()

    # Salva arquivo
    save_path = settings.documents_path / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    config = DocumentCreate(
        obra=obra,
        autor=autor,
        ano=ano,
        idioma=idioma,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        generate_summaries=generate_summaries,
        generate_concept_map=generate_concept_map,
    )

    try:
        doc_id = await indexer.index(save_path, config, db)
    except Exception as e:
        logger.error(f"Erro na indexação: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"document_id": doc_id, "filename": file.filename, "status": "indexed"}


# ── Listagem ───────────────────────────────────────────────────────────────────

@router.get("/", response_model=list[dict])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    obra: str = "",
    autor: str = "",
):
    """Lista todos os documentos indexados, com filtros opcionais."""
    stmt = select(DocumentORM)
    if obra:
        stmt = stmt.where(DocumentORM.obra == obra)
    if autor:
        stmt = stmt.where(DocumentORM.autor == autor)
    stmt = stmt.order_by(DocumentORM.created_at.desc())

    result = await db.execute(stmt)
    docs = result.scalars().all()

    return [
        {
            "id": d.id,
            "filename": d.filename,
            "titulo": d.titulo,
            "autor": d.autor,
            "obra": d.obra,
            "ano": d.ano,
            "status": d.status,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


# ── Detalhes ───────────────────────────────────────────────────────────────────

@router.get("/{doc_id}")
async def get_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Retorna detalhes completos de um documento (metadados, sumários, mapa conceitual)."""
    import json

    result = await db.execute(select(DocumentORM).where(DocumentORM.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento não encontrado")

    return {
        "id": doc.id,
        "filename": doc.filename,
        "filepath": doc.filepath,
        "titulo": doc.titulo,
        "autor": doc.autor,
        "obra": doc.obra,
        "ano": doc.ano,
        "idioma": doc.idioma,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
        "summary_work": doc.summary_work,
        "summary_chapters": json.loads(doc.summary_chapters or "{}"),
        "concept_map": json.loads(doc.concept_map or "[]"),
        "palavras_chave": doc.get_palavras_chave(),
        "conceitos": doc.get_conceitos(),
        "created_at": doc.created_at.isoformat(),
    }


# ── Chunks hierárquicos ────────────────────────────────────────────────────────

@router.get("/{doc_id}/chunks")
async def get_chunks(
    doc_id: str,
    level: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Retorna a hierarquia de chunks de um documento."""
    stmt = select(ChunkORM).where(ChunkORM.document_id == doc_id)
    if level is not None:
        stmt = stmt.where(ChunkORM.level == level)
    stmt = stmt.order_by(ChunkORM.position)

    result = await db.execute(stmt)
    chunks = result.scalars().all()

    return [
        {
            "id": c.id,
            "parent_id": c.parent_id,
            "level": c.level,
            "position": c.position,
            "title": c.title,
            "text": c.text[:500] + "…" if len(c.text) > 500 else c.text,
            "summary": c.summary,
            "obra": c.obra,
            "capitulo": c.capitulo,
            "token_count": c.token_count,
        }
        for c in chunks
    ]


# ── Remoção ────────────────────────────────────────────────────────────────────

@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Remove documento do vector store e do banco."""
    result = await db.execute(select(DocumentORM).where(DocumentORM.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento não encontrado")

    await indexer.delete(doc_id, db)
    return None
