"""
philorag.services.indexer
──────────────────────────
Orquestra o pipeline completo de indexação:

    1. document_processor  →  extrai texto estruturado
    2. chunker             →  divide em hierarquia de chunks
    3. llm_client.embed    →  gera embeddings (via Ollama/ST)
    4. vector_store        →  persiste no ChromaDB
    5. database            →  persiste metadados no SQLite
    6. rag_pipeline        →  gera sumários e mapa conceitual (opcional)
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

from loguru import logger

from app.config import settings
from app.models.document import ChunkStrategy, DocumentCreate, DocumentStatus
from app.services.chunker import ChunkNode, HierarchicalChunker
from app.services.document_processor import DocumentProcessor
from app.services.llm_client import LLMClient
from app.services.rag_pipeline import HierarchicalRAGPipeline
from app.services.vector_store import VectorStore


class IndexingService:
    """
    Serviço de alto nível para indexar um documento.
    
    Uso:
        svc = IndexingService()
        doc_id = await svc.index(filepath, config)
    """

    def __init__(self):
        self.processor = DocumentProcessor()
        self.llm = LLMClient()
        self.vs = VectorStore()
        self.pipeline = HierarchicalRAGPipeline(
            vector_store=self.vs, llm_client=self.llm
        )

    async def index(
        self,
        filepath: str | Path,
        config: DocumentCreate,
        db_session=None,
    ) -> str:
        """
        Indexa um documento completo.
        
        Returns:
            document_id (str UUID)
        """
        path = Path(filepath)
        doc_id = str(uuid4())
        logger.info(f"Iniciando indexação: {path.name} → {doc_id}")

        # ── Etapa 1: extração ─────────────────────────────────────────────────
        extraction = self.processor.process(path)

        if not extraction.sections:
            raise ValueError(f"Nenhuma seção extraída de {path.name}")

        # ── Etapa 2: chunking ─────────────────────────────────────────────────
        chunker = HierarchicalChunker(
            strategy=config.chunk_strategy,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        base_metadata = {
            "obra":   config.obra or extraction.title,
            "autor":  config.autor or extraction.author,
            "ano":    config.ano,
            "idioma": config.idioma or extraction.language or "pt",
        }

        chunks: list[ChunkNode] = chunker.chunk(extraction, doc_id, base_metadata)
        logger.info(f"Chunks gerados: {len(chunks)}")

        # Apenas chunks de nível ≥ 2 recebem embedding
        indexable = [c for c in chunks if c.level.value >= 2]

        # ── Etapa 3: embeddings ───────────────────────────────────────────────
        logger.info(f"Gerando {len(indexable)} embeddings via {settings.embedding_model}…")
        embeddings = await self._embed_chunks(indexable)

        # ── Etapa 4: vector store ─────────────────────────────────────────────
        self.vs.add_chunks(indexable, embeddings, doc_id)

        # ── Etapa 5: sumários e mapa conceitual (opcional / assíncrono) ───────
        summary_work = ""
        summary_chapters: dict[str, str] = {}
        concept_map: list[dict] = []

        if config.generate_summaries:
            logger.info("Gerando sumário da obra…")
            summary_work = await self.pipeline.generate_summary(
                extraction.raw_text[:6000], level="obra"
            )

            # Sumários por capítulo
            chapter_chunks = [c for c in chunks if c.level.value == 2]
            for cap in chapter_chunks[:10]:  # limita para não sobrecarregar CPU
                if cap.text and len(cap.text) > 100:
                    s = await self.pipeline.generate_summary(cap.text, level="capítulo")
                    summary_chapters[cap.title] = s

        if config.generate_concept_map:
            logger.info("Extraindo mapa conceitual…")
            concept_map = await self.pipeline.generate_concept_map(
                extraction.raw_text[:4000],
                obra=config.obra or extraction.title,
            )

        # ── Etapa 6: persistência SQLite ──────────────────────────────────────
        if db_session is not None:
            await self._persist_document(
                db_session, doc_id, path, extraction, chunks,
                base_metadata, summary_work, summary_chapters, concept_map,
                config,
            )

        logger.info(f"Indexação concluída: {doc_id}")
        return doc_id

    async def _embed_chunks(self, chunks: list[ChunkNode]) -> list[list[float]]:
        """Gera embeddings em lotes para evitar timeouts."""
        import asyncio
        BATCH = 8
        embeddings: list[list[float]] = []
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            batch_embs = await asyncio.gather(
                *[self.llm.embed(c.text[:2000]) for c in batch]
            )
            embeddings.extend(batch_embs)
            logger.debug(f"Embeddings: {i + len(batch)}/{len(chunks)}")
        return embeddings

    async def _persist_document(
        self,
        session,
        doc_id: str,
        path: Path,
        extraction,
        chunks: list[ChunkNode],
        metadata: dict,
        summary_work: str,
        summary_chapters: dict[str, str],
        concept_map: list[dict],
        config: DocumentCreate,
    ) -> None:
        """Persiste documento e chunks no SQLite."""
        from app.database.db import ChunkORM, DocumentORM

        doc_orm = DocumentORM(
            id=doc_id,
            filename=path.name,
            filepath=str(path),
            mimetype=extraction.mimetype,
            status=DocumentStatus.INDEXED.value,
            titulo=extraction.title,
            autor=metadata.get("autor", ""),
            ano=config.ano,
            idioma=metadata.get("idioma", "pt"),
            obra=metadata.get("obra", ""),
            summary_work=summary_work,
            summary_chapters=json.dumps(summary_chapters, ensure_ascii=False),
            concept_map=json.dumps(concept_map, ensure_ascii=False),
            chunk_count=len([c for c in chunks if c.level.value >= 3]),
        )
        session.add(doc_orm)

        for chunk in chunks:
            chunk_orm = ChunkORM(
                id=chunk.id,
                document_id=doc_id,
                parent_id=chunk.parent_id,
                level=chunk.level.value,
                position=chunk.position,
                title=chunk.title,
                text=chunk.text,
                markdown=chunk.markdown,
                obra=chunk.metadata.get("obra", ""),
                capitulo=chunk.metadata.get("capitulo", ""),
                secao=chunk.metadata.get("secao", ""),
                conceitos=json.dumps(
                    chunk.metadata.get("conceitos", []), ensure_ascii=False
                ),
                token_count=chunk.token_count,
            )
            session.add(chunk_orm)

        await session.commit()
        logger.debug(f"Documento e {len(chunks)} chunks persistidos no SQLite")

    async def delete(self, document_id: str, db_session=None) -> None:
        """Remove documento do vector store e do SQLite."""
        self.vs.delete_document(document_id)

        if db_session is not None:
            from sqlalchemy import delete
            from app.database.db import ChunkORM, DocumentORM

            await db_session.execute(
                delete(ChunkORM).where(ChunkORM.document_id == document_id)
            )
            await db_session.execute(
                delete(DocumentORM).where(DocumentORM.id == document_id)
            )
            await db_session.commit()
            logger.info(f"Documento {document_id} removido")
