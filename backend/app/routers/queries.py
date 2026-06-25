"""
philorag.routers.queries
──────────────────────────
Endpoints para consulta RAG ao corpus filosófico.

POST /query           →  consulta RAG hierárquica
POST /query/dossier   →  gera dossiê temático
GET  /library/status  →  status da biblioteca (docs indexados, modelos disponíveis)
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.db import DocumentORM, get_db
from app.models.query import (
    Dossier, DossierRequest, DossierSection,
    QueryMode, QueryRequest, RAGResponse,
)
from app.services.llm_client import LLMClient
from app.services.rag_pipeline import HierarchicalRAGPipeline
from app.services.vector_store import VectorStore

router = APIRouter(tags=["queries"])

vs = VectorStore()
llm = LLMClient()
pipeline = HierarchicalRAGPipeline(vector_store=vs, llm_client=llm)


# ── Helpers: carrega contexto hierárquico do SQLite ───────────────────────────

async def _load_context(db: AsyncSession, document_ids: list[str] | None = None) -> tuple:
    """
    Carrega sumários e mapas conceituais do SQLite para injeção hierárquica.
    
    Returns:
        (document_summaries, chunk_parents, concept_maps)
    """
    stmt = select(DocumentORM)
    if document_ids:
        stmt = stmt.where(DocumentORM.id.in_(document_ids))
    result = await db.execute(stmt)
    docs = result.scalars().all()

    document_summaries = {d.id: d.summary_work for d in docs if d.summary_work}
    concept_maps = {
        d.obra: json.loads(d.concept_map or "[]")
        for d in docs
        if d.obra and d.concept_map
    }

    # chunk_parents: para MVP, retorna dicionário vazio
    # (implementação completa requer join chunks→parent)
    chunk_parents: dict = {}

    return document_summaries, chunk_parents, concept_maps


# ── Consulta RAG ───────────────────────────────────────────────────────────────

@router.post("/query", response_model=RAGResponse)
async def query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Consulta RAG hierárquica ao corpus filosófico.
    
    Exemplos:
    - Exegética: "Reconstrua o conceito de animalidade em ZA I"
    - Bibliográfica: "Quais comentadores discutem a crítica ao antropocentrismo?"
    - Comparativa: "Compare Lemm e Cragnolini sobre a animalidade em Nietzsche"
    - Dossiê: ver /query/dossier
    """
    doc_ids = [str(d) for d in request.document_ids] if request.document_ids else None

    document_summaries, chunk_parents, concept_maps = await _load_context(db, doc_ids)

    try:
        response = await pipeline.query(
            request=request,
            document_summaries=document_summaries,
            chunk_parents=chunk_parents,
            concept_maps=concept_maps,
        )
    except Exception as e:
        logger.error(f"Erro no pipeline RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return response


# ── Dossiê temático ────────────────────────────────────────────────────────────

@router.post("/query/dossier", response_model=Dossier)
async def generate_dossier(
    request: DossierRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Gera um dossiê temático completo sobre um conceito/tema.
    
    Estrutura:
    1. Delimitação do conceito
    2. Textos primários
    3. Recepção crítica
    4. Debates em aberto
    5. Bibliografia
    """
    document_summaries, chunk_parents, concept_maps = await _load_context(db)

    sections = []

    # Seção 1: delimitação conceitual
    q1 = QueryRequest(
        question=f"Delimite filosoficamente o conceito de '{request.theme}': "
                 f"definição, contexto teórico, relevância no corpus.",
        mode=QueryMode.EXEGETICAL,
        temperature=0.2,
    )
    r1 = await pipeline.query(q1, document_summaries, chunk_parents, concept_maps)
    sections.append(DossierSection(
        title=f"1. O conceito de '{request.theme}': delimitação filosófica",
        content=r1.answer, sources=r1.citations,
    ))

    # Seção 2: textos primários
    if request.include_primary:
        q2 = QueryRequest(
            question=f"Rastree as ocorrências e desenvolvimento do tema '{request.theme}' "
                     f"nos textos primários do corpus.",
            mode=QueryMode.EXEGETICAL,
            temperature=0.2,
        )
        r2 = await pipeline.query(q2, document_summaries, chunk_parents, concept_maps)
        sections.append(DossierSection(
            title=f"2. Textos primários: '{request.theme}'",
            content=r2.answer, sources=r2.citations,
        ))

    # Seção 3: comentadores
    if request.include_commentators:
        q3 = QueryRequest(
            question=f"Como os comentadores secundários tratam o tema '{request.theme}'? "
                     f"Identifique perspectivas, divergências e contribuições.",
            mode=QueryMode.BIBLIOGRAPHIC,
            temperature=0.3,
        )
        r3 = await pipeline.query(q3, document_summaries, chunk_parents, concept_maps)
        sections.append(DossierSection(
            title=f"3. Recepção crítica: '{request.theme}' nos comentadores",
            content=r3.answer, sources=r3.citations,
        ))

    # Seção 4: debates abertos
    q4 = QueryRequest(
        question=f"Quais os debates interpretativos em aberto sobre '{request.theme}'? "
                 f"Quais lacunas permanecem na literatura especializada?",
        mode=QueryMode.COMPARATIVE,
        temperature=0.4,
    )
    r4 = await pipeline.query(q4, document_summaries, chunk_parents, concept_maps)
    sections.append(DossierSection(
        title=f"4. Debates em aberto e lacunas interpretativas",
        content=r4.answer, sources=r4.citations,
    ))

    # Mapa conceitual do tema
    all_sources = []
    for s in sections:
        all_sources.extend(s.sources)

    obras_map = {}
    for src in all_sources:
        obra = src.metadata_obra
        if obra not in obras_map:
            obras_map[obra] = []
        if src.title not in obras_map[obra]:
            obras_map[obra].append(src.title)

    # Bibliografia simplificada (ABNT-like)
    bibliography = _format_bibliography(all_sources, style=request.citation_style)

    return Dossier(
        theme=request.theme,
        sections=sections,
        bibliography=bibliography,
        concept_map=obras_map,
    )


# ── Status da biblioteca ───────────────────────────────────────────────────────

@router.get("/library/status")
async def library_status(db: AsyncSession = Depends(get_db)):
    """Retorna status geral: documentos, chunks, modelos."""
    result = await db.execute(select(DocumentORM))
    docs = result.scalars().all()

    total_chunks = sum(d.chunk_count for d in docs)
    obras = list({d.obra for d in docs if d.obra})

    ollama_ok = await llm.is_ollama_available()
    models = await llm.list_ollama_models() if ollama_ok else []

    return {
        "documents": len(docs),
        "total_chunks": total_chunks,
        "obras_indexadas": obras,
        "autores": list({d.autor for d in docs if d.autor}),
        "vector_store": {
            "path": str(llm.model),
            "count": vs.count(),
        },
        "ollama": {
            "available": ollama_ok,
            "models": models,
            "active_model": llm.model,
            "embedding_model": llm._st_model if hasattr(llm, "_st_model") else "nomic-embed-text",
        },
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_bibliography(sources, style: str = "abnt") -> list[str]:
    """Formata referências bibliográficas a partir dos chunks recuperados."""
    seen = set()
    refs = []
    for src in sources:
        key = f"{src.metadata_autor}:{src.metadata_obra}"
        if key in seen or not src.metadata_autor:
            continue
        seen.add(key)
        if style == "abnt":
            ref = f"{src.metadata_autor.upper()}. {src.metadata_obra}. {src.metadata_ano}."
        else:  # chicago / apa
            ref = f"{src.metadata_autor}. {src.metadata_obra} ({src.metadata_ano})."
        refs.append(ref)
    return sorted(refs)
