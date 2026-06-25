#!/usr/bin/env python3
"""
Conversor de documentos usando Docling.
Uso: python docling_convert.py <caminho_do_arquivo>
"""

from pathlib import Path
import sys
from docling.document_converter import DocumentConverter

def main():
    if len(sys.argv) < 2:
        print("Uso: python docling_convert.py <caminho_do_arquivo>")
        sys.exit(1)

    source = sys.argv[1]
    if not Path(source).exists():
        print(f"Erro: arquivo '{source}' não encontrado.")
        sys.exit(1)

    converter = DocumentConverter()
    result = converter.convert(source)
    
    # Salva o Markdown em um arquivo .md
    output_file = Path(source).stem + ".md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())
    
    print(f"✅ Conversão concluída! Arquivo salvo em: {output_file}")

if __name__ == "__main__":
    main()