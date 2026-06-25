#!/usr/bin/env python3
"""
Conversor de documentos com Docling - versão interativa.
Suporta: PDF, DOCX, PPTX, XLSX, imagens, HTML, etc.
"""

import os
from pathlib import Path
from docling.document_converter import DocumentConverter

def converter_arquivo(caminho_entrada, caminho_saida=None, formato="markdown"):
    """
    Converte um documento para o formato especificado.
    
    Args:
        caminho_entrada (str): Caminho do arquivo de entrada.
        caminho_saida (str, optional): Caminho do arquivo de saída.
        formato (str): Formato de saída ('markdown', 'json', 'text').
    
    Returns:
        str: Caminho do arquivo gerado.
    """
    entrada = Path(caminho_entrada)
    if not entrada.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_entrada}")
    
    # Define o nome do arquivo de saída se não foi especificado
    if caminho_saida is None:
        sufixo = ".md" if formato == "markdown" else ".json"
        caminho_saida = entrada.parent / (entrada.stem + sufixo)
    else:
        caminho_saida = Path(caminho_saida)
    
    # Converte
    converter = DocumentConverter()
    result = converter.convert(str(entrada))
    
    # Exporta no formato desejado
    if formato == "markdown":
        conteudo = result.document.export_to_markdown()
    elif formato == "json":
        conteudo = result.document.export_to_dict()  # ou .model_dump_json()
    else:
        conteudo = result.document.export_to_text()
    
    # Salva
    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.write(conteudo)
    
    return str(caminho_saida)

def main():
    import sys
    if len(sys.argv) < 2:
        print("Uso: python docling_convert.py <arquivo> [--formato markdown|json|text] [--saida caminho]")
        print("\nExemplos:")
        print("  python docling_convert.py documento.pdf")
        print("  python docling_convert.py documento.pdf --formato json --saida saida.json")
        sys.exit(1)
    
    entrada = sys.argv[1]
    formato = "markdown"
    saida = None
    
    # Processa argumentos opcionais
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--formato" and i + 1 < len(sys.argv):
            formato = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--saida" and i + 1 < len(sys.argv):
            saida = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    try:
        arquivo_gerado = converter_arquivo(entrada, saida, formato)
        print(f"✅ Conversão concluída! Arquivo salvo em: {arquivo_gerado}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()