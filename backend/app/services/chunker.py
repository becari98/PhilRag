"""
philorag.services.chunker
──────────────────────────
Chunking hierárquico para corpora filosóficos.

Inovação central do PhiloRAG: o chunking não é apenas divisão por tokens,
mas preservação da estrutura retórica e argumentativa do texto.

Hierarquia:
    Level 1  →  Obra completa (1 chunk de sumário)
    Level 2  →  Capítulo / Discurso / Parte
    Level 3  →  Parágrafo (unidade primária de busca vetorial)
    Level 4  →  Sentença (granularidade fina, opcional)

Estratégias:
    HIERARCHICAL  →  segue estrutura detectada na extração
    TOKEN         →  divisão por número de tokens (overlapping)
    APHORISM      →  para Nietzsche: divide por §/No. numerados
    PARAGRAPH     →  divide por parágrafos (\n\n)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from app.models.document import ChunkLevel, ChunkStrategy
from app.services.document_processor import ExtractionResult, ExtractedSection


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class ChunkNode:
    """Nó na árvore hierárquica de chunks."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    level: ChunkLevel = ChunkLevel.PARAGRAPH
    position: int = 0
    title: str = ""
    text: str = ""
    markdown: str = ""
    token_count: int = 0
    metadata: dict = field(default_factory=dict)
    children: list["ChunkNode"] = field(default_factory=list)

    def to_markdown(self) -> str:
        prefix = "#" * self.level.value
        if self.title:
            return f"{prefix} {self.title}\n\n{self.text}"
        return self.text


# ── Chunker ───────────────────────────────────────────────────────────────────

class HierarchicalChunker:
    """
    Produz árvore de chunks a partir de ExtractionResult.
    
    O resultado é uma lista plana de ChunkNode com referências
    parent_id para reconstrução da hierarquia.
    """

    # Padrão para aforismos numerados (§ 1, No. 1, Aphorism 1, etc.)
    APHORISM_RE = re.compile(
        r"(?:^|\n)\s*(?:§|No\.|Aphorism|Aforismo|Spruch)\s*(\d+)",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.HIERARCHICAL,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_tokens: int = 20,
        sentence_level: bool = False,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        self.sentence_level = sentence_level

    def chunk(
        self,
        extraction: ExtractionResult,
        document_id: str,
        base_metadata: dict | None = None,
    ) -> list[ChunkNode]:
        """Ponto de entrada. Retorna lista plana ordenada por posição."""
        metadata = base_metadata or {}

        # Nó raiz: obra completa (level 1)
        root = ChunkNode(
            level=ChunkLevel.WORK,
            position=0,
            title=extraction.title or "Obra",
            text=extraction.raw_text[:2000] + "…" if len(extraction.raw_text) > 2000 else extraction.raw_text,
            metadata={**metadata, "obra": extraction.title, "autor": extraction.author},
        )

        if self.strategy == ChunkStrategy.HIERARCHICAL:
            nodes = self._hierarchical(root, extraction.sections, metadata)
        elif self.strategy == ChunkStrategy.APHORISM:
            nodes = self._aphorism(root, extraction.raw_text, metadata)
        elif self.strategy == ChunkStrategy.TOKEN:
            nodes = self._token_based(root, extraction.raw_text, metadata)
        else:
            nodes = self._paragraph_based(root, extraction.raw_text, metadata)

        # Calcula token_count aproximado (4 chars ≈ 1 token)
        for node in nodes:
            node.token_count = len(node.text) // 4
            node.markdown = node.to_markdown()

        logger.info(
            f"Chunking {self.strategy.value}: "
            f"{len([n for n in nodes if n.level == ChunkLevel.CHAPTER])} capítulos, "
            f"{len([n for n in nodes if n.level == ChunkLevel.PARAGRAPH])} parágrafos"
        )
        return nodes

    # ── Estratégias ───────────────────────────────────────────────────────────

    def _hierarchical(
        self,
        root: ChunkNode,
        sections: list[ExtractedSection],
        metadata: dict,
    ) -> list[ChunkNode]:
        """Segue a estrutura detectada pelo DocumentProcessor."""
        nodes: list[ChunkNode] = [root]
        current_chapter: ChunkNode | None = None
        para_position = 0

        for section in sections:
            if section.level == 2:  # Capítulo
                current_chapter = ChunkNode(
                    parent_id=root.id,
                    level=ChunkLevel.CHAPTER,
                    position=len(nodes),
                    title=section.title,
                    text=section.text,
                    metadata={**metadata, "capitulo": section.title},
                )
                nodes.append(current_chapter)
                para_position = 0

            elif section.level == 3:  # Parágrafo / corpo
                parent_id = current_chapter.id if current_chapter else root.id
                parent_title = current_chapter.title if current_chapter else ""

                # Sub-divide por parágrafos se o texto for muito longo
                sub_paras = self._split_paragraphs(section.text)
                for para in sub_paras:
                    if len(para.strip()) < 60:  # filtra fragmentos
                        continue
                    node = ChunkNode(
                        parent_id=parent_id,
                        level=ChunkLevel.PARAGRAPH,
                        position=len(nodes),
                        title=parent_title,
                        text=para.strip(),
                        metadata={**metadata, "capitulo": parent_title, "secao": section.title},
                    )
                    nodes.append(node)
                    para_position += 1

                    # Nível sentença (opcional)
                    if self.sentence_level:
                        for sent in self._split_sentences(para):
                            if len(sent.strip()) > 30:
                                nodes.append(ChunkNode(
                                    parent_id=node.id,
                                    level=ChunkLevel.SENTENCE,
                                    position=len(nodes),
                                    text=sent.strip(),
                                    metadata=node.metadata.copy(),
                                ))

        return nodes

    def _aphorism(
        self,
        root: ChunkNode,
        text: str,
        metadata: dict,
    ) -> list[ChunkNode]:
        """
        Divide por aforismos numerados — ideal para Nietzsche (ABM, GC, A…).
        
        Ex.: "§ 1 Über die Vorurteile der Philosophen…"
        """
        nodes: list[ChunkNode] = [root]
        splits = self.APHORISM_RE.split(text)

        # splits: [pre-text, numero, conteúdo, numero, conteúdo, ...]
        if len(splits) < 3:
            logger.warning("Padrão de aforismo não encontrado; usando estratégia por parágrafo")
            return self._paragraph_based(root, text, metadata)

        # Primeiro bloco (antes do §1) pode ser prefácio
        if splits[0].strip():
            nodes.append(ChunkNode(
                parent_id=root.id,
                level=ChunkLevel.CHAPTER,
                position=1,
                title="Prefácio",
                text=splits[0].strip(),
                metadata={**metadata, "capitulo": "Prefácio"},
            ))

        i = 1
        while i < len(splits) - 1:
            number = splits[i].strip()
            content = splits[i + 1].strip() if i + 1 < len(splits) else ""
            if content:
                title = f"§ {number}"
                aph_node = ChunkNode(
                    parent_id=root.id,
                    level=ChunkLevel.CHAPTER,
                    position=len(nodes),
                    title=title,
                    text=content,
                    metadata={**metadata, "capitulo": title},
                )
                nodes.append(aph_node)

                # Parágrafos dentro do aforismo
                for sub in self._split_paragraphs(content):
                    if len(sub.strip()) > 60:
                        nodes.append(ChunkNode(
                            parent_id=aph_node.id,
                            level=ChunkLevel.PARAGRAPH,
                            position=len(nodes),
                            title=title,
                            text=sub.strip(),
                            metadata=aph_node.metadata.copy(),
                        ))
            i += 2

        return nodes

    def _token_based(
        self,
        root: ChunkNode,
        text: str,
        metadata: dict,
    ) -> list[ChunkNode]:
        """Divisão clássica por tokens com overlap."""
        nodes: list[ChunkNode] = [root]
        words = text.split()
        step = max(1, self.chunk_size - self.chunk_overlap)

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            if len(chunk_text) < 60:
                continue
            nodes.append(ChunkNode(
                parent_id=root.id,
                level=ChunkLevel.PARAGRAPH,
                position=len(nodes),
                text=chunk_text,
                metadata=metadata.copy(),
            ))

        return nodes

    def _paragraph_based(
        self,
        root: ChunkNode,
        text: str,
        metadata: dict,
    ) -> list[ChunkNode]:
        """Divide por parágrafos (\n\n)."""
        nodes: list[ChunkNode] = [root]
        paras = self._split_paragraphs(text)

        for i, para in enumerate(paras):
            if len(para.strip()) < 60:
                continue
            nodes.append(ChunkNode(
                parent_id=root.id,
                level=ChunkLevel.PARAGRAPH,
                position=len(nodes),
                text=para.strip(),
                metadata=metadata.copy(),
            ))

        return nodes

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _split_paragraphs(text: str, max_chars: int = 1500) -> list[str]:
        """Divide por \n\n; subdivide parágrafos muito longos."""
        raw = re.split(r"\n{2,}", text)
        result = []
        for para in raw:
            para = para.strip()
            if not para:
                continue
            if len(para) <= max_chars:
                result.append(para)
            else:
                # Subdivide por sentença
                sents = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sents:
                    if len(current) + len(sent) > max_chars and current:
                        result.append(current.strip())
                        current = sent
                    else:
                        current += " " + sent
                if current.strip():
                    result.append(current.strip())
        return result

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Divide por sentenças (pontuação final)."""
        return re.split(r"(?<=[.!?])\s+", text)
