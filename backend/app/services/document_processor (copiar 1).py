"""
philorag.services.document_processor
─────────────────────────────────────
Extração de texto de múltiplos formatos com detecção estrutural.

Formatos suportados:
    PDF nativo    → pymupdf (fitz)
    PDF escaneado → pymupdf + pytesseract
    DOCX          → python-docx
    EPUB          → ebooklib
    TXT / MD      → leitura direta
    HTML          → html2text / BeautifulSoup

Saída padronizada:
    List[ExtractedSection]  →  cada seção com nível, título e texto
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ExtractedSection:
    """Seção estrutural extraída de um documento."""
    level: int          # 1=obra, 2=capítulo, 3=parágrafo
    title: str
    text: str
    position: int       # ordem no documento
    page_start: int = 0
    page_end: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Resultado completo da extração de um documento."""
    filepath: str
    mimetype: str
    title: str
    author: str
    language: str
    sections: list[ExtractedSection] = field(default_factory=list)
    raw_text: str = ""
    page_count: int = 0
    ocr_used: bool = False


# ── Processor ─────────────────────────────────────────────────────────────────

class DocumentProcessor:
    """
    Extrai texto estruturado de documentos acadêmicos.
    
    A extração tenta identificar:
    - Títulos de capítulos/seções (heurística de tamanho + posição)
    - Estrutura hierárquica implícita nos PDFs
    - Parágrafos coerentes
    """

    MIME_MAP = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".epub": "application/epub+zip",
        ".txt":  "text/plain",
        ".md":   "text/markdown",
        ".html": "text/html",
        ".htm":  "text/html",
        ".odt":  "application/vnd.oasis.opendocument.text",
    }

    def process(self, filepath: str | Path, ocr_fallback: bool = True) -> ExtractionResult:
        """Ponto de entrada principal. Detecta formato e delega."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        suffix = path.suffix.lower()
        mimetype = self.MIME_MAP.get(suffix, "application/octet-stream")

        logger.info(f"Processando {path.name} ({mimetype})")

        dispatch = {
            ".pdf":  self._process_pdf,
            ".docx": self._process_docx,
            ".epub": self._process_epub,
            ".txt":  self._process_plaintext,
            ".md":   self._process_markdown,
            ".html": self._process_html,
            ".htm":  self._process_html,
        }

        handler = dispatch.get(suffix)
        if handler is None:
            raise ValueError(f"Formato não suportado: {suffix}")

        result = handler(path, ocr_fallback)
        result.mimetype = mimetype
        result.filepath = str(path)

        logger.info(f"Extraídas {len(result.sections)} seções de {path.name}")
        return result

    # ── PDF ───────────────────────────────────────────────────────────────────

    def _process_pdf(self, path: Path, ocr_fallback: bool = True) -> ExtractionResult:
        try:
            import fitz  # pymupdf
        except ImportError:
            raise ImportError("pymupdf não instalado: pip install pymupdf")

        doc = fitz.open(str(path))
        result = ExtractionResult(
            filepath=str(path),
            mimetype="application/pdf",
            title=doc.metadata.get("title", path.stem),
            author=doc.metadata.get("author", ""),
            language=doc.metadata.get("language", ""),
            page_count=len(doc),
        )

        all_blocks: list[dict] = []
        needs_ocr = True

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") == 0:  # text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                needs_ocr = False
                                all_blocks.append({
                                    "text": text,
                                    "size": span.get("size", 12),
                                    "flags": span.get("flags", 0),  # bold=16, italic=2
                                    "page": page_num + 1,
                                })

        # Se o PDF for escaneado e OCR disponível
        if needs_ocr and ocr_fallback:
            logger.info("PDF escaneado detectado — tentando OCR com Tesseract")
            result = self._ocr_pdf(path, doc, result)
            return result

        doc.close()

        # Agrupa blocos em seções hierárquicas
        result.sections = self._group_pdf_blocks(all_blocks)
        result.raw_text = "\n\n".join(s.text for s in result.sections)
        return result

    def _group_pdf_blocks(self, blocks: list[dict]) -> list[ExtractedSection]:
        """
        Heurística para identificar títulos vs. corpo em PDFs.
        
        Títulos: fonte ≥ 14pt OU texto em bold curto (< 80 chars)
        Corpo: demais blocos
        """
        if not blocks:
            return []

        # Determina tamanho de fonte dominante (modo)
        sizes = [b["size"] for b in blocks]
        body_size = max(set(sizes), key=sizes.count)
        heading_threshold = body_size * 1.15  # 15% maior que o corpo

        sections: list[ExtractedSection] = []
        current_chapter_title = ""
        current_texts: list[str] = []
        position = 0
        current_page = 1

        for block in blocks:
            size = block["size"]
            flags = block["flags"]
            text = block["text"]
            is_bold = bool(flags & 16)
            is_heading = (
                size >= heading_threshold
                or (is_bold and len(text) < 120 and len(text) > 2)
            )

            if is_heading and len(text.split()) >= 2:
                # Flush parágrafo atual
                if current_texts:
                    sections.append(ExtractedSection(
                        level=3,
                        title=current_chapter_title,
                        text=self._clean_text(" ".join(current_texts)),
                        position=position,
                        page_start=current_page,
                    ))
                    position += 1
                    current_texts = []

                # Novo capítulo
                current_chapter_title = text
                sections.append(ExtractedSection(
                    level=2,
                    title=text,
                    text=text,
                    position=position,
                    page_start=block["page"],
                ))
                position += 1
                current_page = block["page"]
            else:
                current_texts.append(text)

        # Flush final
        if current_texts:
            sections.append(ExtractedSection(
                level=3,
                title=current_chapter_title,
                text=self._clean_text(" ".join(current_texts)),
                position=position,
            ))

        return sections

    def _ocr_pdf(self, path: Path, doc, result: ExtractionResult) -> ExtractionResult:
        """OCR com Tesseract via pymupdf."""
        try:
            import pytesseract
            from PIL import Image
            import io
        except ImportError:
            raise ImportError("pytesseract/Pillow não instalados para OCR")

        result.ocr_used = True
        sections = []
        position = 0

        for page_num, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="por+deu+eng")
            if text.strip():
                sections.append(ExtractedSection(
                    level=3,
                    title=f"Página {page_num + 1}",
                    text=self._clean_text(text),
                    position=position,
                    page_start=page_num + 1,
                ))
                position += 1

        doc.close()
        result.sections = sections
        result.raw_text = "\n\n".join(s.text for s in sections)
        return result

    # ── DOCX ──────────────────────────────────────────────────────────────────

    def _process_docx(self, path: Path, *_) -> ExtractionResult:
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("python-docx não instalado: pip install python-docx")

        doc = DocxDocument(str(path))
        result = ExtractionResult(
            filepath=str(path), mimetype="", title=path.stem,
            author="", language="",
        )

        sections = []
        position = 0
        current_chapter = ""
        current_texts: list[str] = []

        for para in doc.paragraphs:
            style = para.style.name.lower()
            text = para.text.strip()
            if not text:
                continue

            if "heading" in style or style.startswith("título"):
                if current_texts:
                    sections.append(ExtractedSection(
                        level=3, title=current_chapter,
                        text=self._clean_text("\n".join(current_texts)),
                        position=position,
                    ))
                    position += 1
                    current_texts = []
                level = 2 if "1" in style else 2
                current_chapter = text
                sections.append(ExtractedSection(
                    level=level, title=text, text=text, position=position,
                ))
                position += 1
            else:
                current_texts.append(text)

        if current_texts:
            sections.append(ExtractedSection(
                level=3, title=current_chapter,
                text=self._clean_text("\n".join(current_texts)),
                position=position,
            ))

        result.sections = sections
        result.raw_text = "\n\n".join(s.text for s in sections)
        return result

    # ── EPUB ──────────────────────────────────────────────────────────────────

    def _process_epub(self, path: Path, *_) -> ExtractionResult:
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("ebooklib/beautifulsoup4 não instalados")

        book = epub.read_epub(str(path))
        result = ExtractionResult(
            filepath=str(path), mimetype="", title=path.stem,
            author="", language="",
        )

        sections = []
        position = 0

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            heading = soup.find(["h1", "h2", "h3"])
            title = heading.get_text(strip=True) if heading else ""
            text = soup.get_text(separator="\n", strip=True)
            if text:
                level = 2 if heading and heading.name in ("h1", "h2") else 3
                sections.append(ExtractedSection(
                    level=level, title=title,
                    text=self._clean_text(text), position=position,
                ))
                position += 1

        result.sections = sections
        result.raw_text = "\n\n".join(s.text for s in sections)
        return result

    # ── TXT / Markdown / HTML ────────────────────────────────────────────────

    def _process_plaintext(self, path: Path, *_) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        return self._text_to_result(path, text)

    def _process_markdown(self, path: Path, *_) -> ExtractionResult:
        try:
            import markdown
            from bs4 import BeautifulSoup
        except ImportError:
            return self._process_plaintext(path)

        md_text = path.read_text(encoding="utf-8", errors="replace")
        html = markdown.markdown(md_text, extensions=["toc", "tables"])
        soup = BeautifulSoup(html, "html.parser")

        sections = []
        position = 0
        current_title = ""
        current_texts: list[str] = []

        for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol"]):
            if elem.name in ("h1", "h2", "h3", "h4"):
                if current_texts:
                    sections.append(ExtractedSection(
                        level=3, title=current_title,
                        text=self._clean_text("\n".join(current_texts)),
                        position=position,
                    ))
                    position += 1
                    current_texts = []
                current_title = elem.get_text(strip=True)
                level = 2 if elem.name in ("h1", "h2") else 3
                sections.append(ExtractedSection(
                    level=level, title=current_title, text=current_title,
                    position=position,
                ))
                position += 1
            else:
                t = elem.get_text(separator=" ", strip=True)
                if t:
                    current_texts.append(t)

        if current_texts:
            sections.append(ExtractedSection(
                level=3, title=current_title,
                text=self._clean_text("\n".join(current_texts)),
                position=position,
            ))

        result = ExtractionResult(
            filepath=str(path), mimetype="", title=path.stem,
            author="", language="",
            sections=sections,
        )
        result.raw_text = "\n\n".join(s.text for s in sections)
        return result

    def _process_html(self, path: Path, *_) -> ExtractionResult:
        try:
            import html2text
        except ImportError:
            from bs4 import BeautifulSoup
            html = path.read_text(encoding="utf-8", errors="replace")
            text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
            return self._text_to_result(path, text)

        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        html_content = path.read_text(encoding="utf-8", errors="replace")
        md_text = h.handle(html_content)
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(md_text)
        result = self._process_markdown(tmp)
        tmp.unlink(missing_ok=True)
        result.filepath = str(path)
        return result

    def _text_to_result(self, path: Path, text: str) -> ExtractionResult:
        """Converte texto plano em seções por parágrafos."""
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        sections = [
            ExtractedSection(level=3, title="", text=self._clean_text(p), position=i)
            for i, p in enumerate(paragraphs)
        ]
        return ExtractionResult(
            filepath=str(path), mimetype="", title=path.stem,
            author="", language="",
            sections=sections, raw_text=text,
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normaliza whitespace e remove artefatos comuns de PDF."""
        text = re.sub(r"-\n(\w)", r"\1", text)          # hifenização
        text = re.sub(r"\n(?=[a-záéíóúàãõâêîôûçüñ])", " ", text, flags=re.I)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
