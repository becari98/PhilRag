"""
philorag.services.document_processor
─────────────────────────────────────
Extração de texto de múltiplos formatos com detecção estrutural.

Formatos suportados (nativos):  PDF, DOCX, EPUB, TXT, MD, HTML, ODT.
Com Docling: suporte estendido para PDFs com layout complexo, OCR, tabelas e imagens.

Estratégia:
    1. Tenta processar com Docling (se instalado) – recomendado para PDFs acadêmicos.
    2. Fallback para o processador legado (pymupdf, python-docx, etc.).
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
    Extrai texto estruturado usando Docling (preferencial) ou fallback.
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

    def __init__(self, prefer_docling: bool = True):
        """
        Args:
            prefer_docling: Se True, tenta Docling primeiro (para PDFs).
        """
        self.prefer_docling = prefer_docling
        self._docling_available = self._check_docling()

    @staticmethod
    def _check_docling() -> bool:
        try:
            import docling  # noqa
            return True
        except ImportError:
            return False

    def process(self, filepath: str | Path, ocr_fallback: bool = True) -> ExtractionResult:
        """Ponto de entrada principal. Detecta formato e delega."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        suffix = path.suffix.lower()
        mimetype = self.MIME_MAP.get(suffix, "application/octet-stream")

        logger.info(f"Processando {path.name} ({mimetype})")

        # ── Tenta Docling para PDFs (e outros suportados) ──────────────────
        if self.prefer_docling and self._docling_available and suffix in (".pdf", ".docx", ".pptx", ".xlsx"):
            try:
                result = self._process_with_docling(path)
                result.mimetype = mimetype
                result.filepath = str(path)
                logger.info(f"Docling: extraídas {len(result.sections)} seções de {path.name}")
                return result
            except Exception as e:
                logger.warning(f"Docling falhou para {path.name}: {e}. Usando fallback.")

        # ── Fallback legado ────────────────────────────────────────────────
        dispatch = {
            ".pdf":  self._process_pdf_legacy,
            ".docx": self._process_docx_legacy,
            ".epub": self._process_epub_legacy,
            ".txt":  self._process_plaintext_legacy,
            ".md":   self._process_markdown_legacy,
            ".html": self._process_html_legacy,
            ".htm":  self._process_html_legacy,
            ".odt":  self._process_odt_legacy,
        }

        handler = dispatch.get(suffix)
        if handler is None:
            raise ValueError(f"Formato não suportado: {suffix}")

        result = handler(path, ocr_fallback)
        result.mimetype = mimetype
        result.filepath = str(path)

        logger.info(f"Fallback: extraídas {len(result.sections)} seções de {path.name}")
        return result

    # ── DOCLING ──────────────────────────────────────────────────────────────

    def _process_with_docling(self, path: Path) -> ExtractionResult:
        """Processa com Docling e converte para ExtractionResult."""
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat

        converter = DocumentConverter()
        # Para PDFs, ativa OCR automático se necessário
        if path.suffix.lower() == ".pdf":
            converter = DocumentConverter(
                pipeline_options={"ocr_strategy": "auto", "table_structure": True}
            )

        result = converter.convert(str(path))
        doc = result.document

        # Extrai Markdown completo
        md = doc.export_to_markdown()

        # Extrai metadados
        meta = doc.meta
        title = meta.get("title", path.stem)
        author = meta.get("author", "")
        language = meta.get("language", "")

        # Extrai seções a partir de headings no Markdown
        sections = []
        position = 0
        current_level = 1
        current_title = ""
        current_text = []

        lines = md.split('\n')
        for line in lines:
            if line.startswith('#'):
                # Finaliza seção anterior
                if current_text:
                    sections.append(ExtractedSection(
                        level=current_level,
                        title=current_title,
                        text='\n'.join(current_text).strip(),
                        position=position,
                        metadata={"heading_level": current_level}
                    ))
                    position += 1
                    current_text = []

                # Nova seção
                heading_level = len(line) - len(line.lstrip('#'))
                # Mapeia # → level 2 (capítulo), ## → level 3 (seção), etc.
                current_level = min(heading_level + 1, 4)
                current_title = line.lstrip('#').strip()
                current_text.append(current_title)  # título também como conteúdo
            else:
                current_text.append(line)

        # Última seção
        if current_text:
            sections.append(ExtractedSection(
                level=current_level,
                title=current_title,
                text='\n'.join(current_text).strip(),
                position=position,
            ))

        # Se não houver headings, cria uma única seção com todo o texto
        if not sections:
            sections.append(ExtractedSection(
                level=1,
                title=title,
                text=md,
                position=0,
            ))

        # Contagem de páginas (se disponível)
        page_count = len(doc.pages) if hasattr(doc, 'pages') else 0

        # Verifica se houve OCR (Docling indica isso)
        ocr_used = getattr(doc, 'ocr_used', False)

        return ExtractionResult(
            filepath=str(path),
            mimetype="",
            title=title,
            author=author,
            language=language,
            sections=sections,
            raw_text=md,
            page_count=page_count,
            ocr_used=ocr_used,
        )

    # ── FALLBACK LEGACY ─────────────────────────────────────────────────────

    def _process_pdf_legacy(self, path: Path, ocr_fallback: bool = True) -> ExtractionResult:
        """Versão original com pymupdf + Tesseract."""
        try:
            import fitz
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
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                needs_ocr = False
                                all_blocks.append({
                                    "text": text,
                                    "size": span.get("size", 12),
                                    "flags": span.get("flags", 0),
                                    "page": page_num + 1,
                                })

        if needs_ocr and ocr_fallback:
            logger.info("PDF escaneado detectado — tentando OCR com Tesseract")
            return self._ocr_pdf_legacy(path, doc, result)

        doc.close()
        result.sections = self._group_pdf_blocks_legacy(all_blocks)
        result.raw_text = "\n\n".join(s.text for s in result.sections)
        return result

    def _group_pdf_blocks_legacy(self, blocks: list[dict]) -> list[ExtractedSection]:
        """Heurística original para identificar títulos."""
        if not blocks:
            return []

        sizes = [b["size"] for b in blocks]
        body_size = max(set(sizes), key=sizes.count)
        heading_threshold = body_size * 1.15

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
            is_heading = (size >= heading_threshold) or (is_bold and len(text) < 120 and len(text) > 2)

            if is_heading and len(text.split()) >= 2:
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

        if current_texts:
            sections.append(ExtractedSection(
                level=3,
                title=current_chapter_title,
                text=self._clean_text(" ".join(current_texts)),
                position=position,
            ))

        return sections

    def _ocr_pdf_legacy(self, path: Path, doc, result: ExtractionResult) -> ExtractionResult:
        """OCR com Tesseract (fallback)."""
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

    def _process_docx_legacy(self, path: Path, *_) -> ExtractionResult:
        """Original DOCX com python-docx."""
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
                current_chapter = text
                sections.append(ExtractedSection(
                    level=2, title=text, text=text, position=position,
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

    def _process_epub_legacy(self, path: Path, *_) -> ExtractionResult:
        """Original EPUB com ebooklib."""
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

    def _process_plaintext_legacy(self, path: Path, *_) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        return self._text_to_result_legacy(path, text)

    def _process_markdown_legacy(self, path: Path, *_) -> ExtractionResult:
        try:
            import markdown
            from bs4 import BeautifulSoup
        except ImportError:
            return self._process_plaintext_legacy(path)

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

    def _process_html_legacy(self, path: Path, *_) -> ExtractionResult:
        try:
            import html2text
        except ImportError:
            from bs4 import BeautifulSoup
            html = path.read_text(encoding="utf-8", errors="replace")
            text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
            return self._text_to_result_legacy(path, text)

        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        html_content = path.read_text(encoding="utf-8", errors="replace")
        md_text = h.handle(html_content)
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(md_text)
        result = self._process_markdown_legacy(tmp)
        tmp.unlink(missing_ok=True)
        result.filepath = str(path)
        return result

    def _process_odt_legacy(self, path: Path, *_) -> ExtractionResult:
        """ODT via texto bruto (fallback)."""
        try:
            # Tenta extrair com zipfile/odfpy? Por simplicidade, lê como zip e extrai content.xml.
            import zipfile
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("zipfile necessário para ODT")

        text_parts = []
        with zipfile.ZipFile(path, 'r') as zf:
            with zf.open('content.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                ns = {'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0',
                      'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
                for elem in root.findall('.//text:p', ns):
                    if elem.text:
                        text_parts.append(elem.text)
        text = '\n\n'.join(text_parts)
        return self._text_to_result_legacy(path, text)

    def _text_to_result_legacy(self, path: Path, text: str) -> ExtractionResult:
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
        text = re.sub(r"-\n(\w)", r"\1", text)
        text = re.sub(r"\n(?=[a-záéíóúàãõâêîôûçüñ])", " ", text, flags=re.I)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()