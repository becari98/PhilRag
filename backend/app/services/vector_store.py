"""
philorag.services.vector_store
───────────────────────────────
Interface ChromaDB com suporte a filtragem por metadados filosóficos.

Cada chunk é armazenado com:
    - embedding vetorial
    - metadados: obra, autor, capítulo, nível, document_id

Isso permite buscas filtradas por:
    - obra específica ("ZA", "ABM", "EH")
    - nível hierárquico
    - autor / período
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from app.config import settings
from app.services.chunker import ChunkNode


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    text: str
    score: float           # distância coseno (menor = mais similar)
    level: int
    title: str
    obra: str
    autor: str
    capitulo: str
    secao: str
    metadata: dict


# ── Vector store ──────────────────────────────────────────────────────────────

class VectorStore:
    """
    Wrapper sobre ChromaDB.
    
    Usa uma única collection com filtragem por metadados.
    Evita múltiplas collections por obra para simplificar cross-corpus queries.
    """

    def __init__(self):
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError("chromadb não instalado: pip install chromadb")

        self._client = chromadb.PersistentClient(
            path=str(settings.vector_store_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB: collection '{settings.chroma_collection_name}' "
            f"({self._collection.count()} documentos)"
        )
        return self._collection

    # ── Indexação ─────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[ChunkNode],
        embeddings: list[list[float]],
        document_id: str,
    ) -> None:
        """Indexa chunks com embeddings no ChromaDB."""
        col = self._get_collection()

        # Filtra chunks de nível 1 (obra completa — não indexados como vetores)
        indexable = [
            (c, e) for c, e in zip(chunks, embeddings)
            if c.level.value >= 2  # capítulo e abaixo
        ]
        if not indexable:
            return

        ids, docs, metas, embs = [], [], [], []
        for chunk, emb in indexable:
            ids.append(chunk.id)
            docs.append(chunk.text[:8000])  # ChromaDB limit
            metas.append({
                "document_id": document_id,
                "level":       chunk.level.value,
                "title":       chunk.title[:200],
                "obra":        chunk.metadata.get("obra", "")[:100],
                "autor":       chunk.metadata.get("autor", "")[:100],
                "capitulo":    chunk.metadata.get("capitulo", "")[:200],
                "secao":       chunk.metadata.get("secao", "")[:200],
                "token_count": chunk.token_count,
            })
            embs.append(emb)

        # Upsert (idempotente)
        col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        logger.info(f"Indexados {len(ids)} chunks (document_id={document_id})")

    def delete_document(self, document_id: str) -> None:
        """Remove todos os chunks de um documento."""
        col = self._get_collection()
        col.delete(where={"document_id": document_id})
        logger.info(f"Removidos chunks do documento {document_id}")

    # ── Busca ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        document_ids: list[str] | None = None,
        obra: str = "",
        autor: str = "",
        level: int | None = None,
    ) -> list[SearchResult]:
        """
        Busca semântica com filtros opcionais.
        
        Filtros combinados com AND no ChromaDB where clause.
        """
        col = self._get_collection()

        where: dict = {}
        conditions = []

        if document_ids:
            conditions.append({"document_id": {"$in": document_ids}})
        if obra:
            conditions.append({"obra": {"$eq": obra}})
        if autor:
            conditions.append({"autor": {"$eq": autor}})
        if level is not None:
            conditions.append({"level": {"$eq": level}})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, col.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        try:
            results = col.query(**kwargs)
        except Exception as e:
            logger.error(f"Erro na busca ChromaDB: {e}")
            return []

        search_results = []
        for i, (doc_id, doc, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            search_results.append(SearchResult(
                chunk_id=doc_id,
                document_id=meta.get("document_id", ""),
                text=doc,
                score=1.0 - float(dist),   # distância coseno → similaridade
                level=meta.get("level", 3),
                title=meta.get("title", ""),
                obra=meta.get("obra", ""),
                autor=meta.get("autor", ""),
                capitulo=meta.get("capitulo", ""),
                secao=meta.get("secao", ""),
                metadata=meta,
            ))

        return search_results

    def count(self, document_id: str | None = None) -> int:
        col = self._get_collection()
        if document_id:
            return col.count()  # filter not directly available in count()
        return col.count()

    # ── Reranking semântico ───────────────────────────────────────────────────

    @staticmethod
    def rerank(
        query: str,
        results: list[SearchResult],
        top_k: int = 8,
    ) -> list[SearchResult]:
        """
        Reranking leve por sobreposição de termos (BM25-like, sem dependências extra).
        Para reranking com cross-encoder, integrar sentence-transformers CrossEncoder.
        """
        query_terms = set(query.lower().split())

        def score(r: SearchResult) -> float:
            text_lower = r.text.lower()
            term_overlap = sum(1 for t in query_terms if t in text_lower)
            return r.score + 0.05 * term_overlap   # boost por presença de termos

        return sorted(results, key=score, reverse=True)[:top_k]
