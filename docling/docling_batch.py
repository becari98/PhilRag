#!/usr/bin/env python3
"""
Processador de pastas com Docling.
Converte todos os documentos suportados dentro de uma pasta (e subpastas).

Uso: python docling_batch.py <pasta_entrada> [--output pasta_saida] [--format md|json|text] [--extensoes .pdf .docx ...]
"""

import os
import sys
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from docling.document_converter import DocumentConverter

# Extensões suportadas pelo Docling
EXTENSOES_SUPORTADAS = {
    '.pdf', '.docx', '.pptx', '.xlsx', '.odt', '.odp', '.ods',
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
    '.html', '.htm', '.xml', '.txt', '.md'
}

def converter_arquivo(entrada: Path, saida_dir: Path, formato: str = "markdown") -> Path | None:
    """
    Converte um único arquivo com Docling.
    
    Args:
        entrada: Caminho do arquivo de entrada.
        saida_dir: Diretório onde salvar o arquivo convertido.
        formato: 'markdown', 'json' ou 'text'.
    
    Returns:
        Caminho do arquivo gerado, ou None se falhar.
    """
    try:
        # Cria o diretório de saída se não existir
        saida_dir.mkdir(parents=True, exist_ok=True)
        
        # Define o nome do arquivo de saída (mesmo nome, extensão diferente)
        if formato == "markdown":
            sufixo = ".md"
        elif formato == "json":
            sufixo = ".json"
        else:
            sufixo = ".txt"
        
        saida = saida_dir / (entrada.stem + sufixo)
        
        # Se o arquivo já existe, pula (opcional)
        if saida.exists():
            return None
        
        # Converte
        converter = DocumentConverter()
        result = converter.convert(str(entrada))
        
        # Exporta no formato desejado
        if formato == "markdown":
            conteudo = result.document.export_to_markdown()
        elif formato == "json":
            conteudo = result.document.model_dump_json(indent=2)
        else:
            conteudo = result.document.export_to_text()
        
        # Salva
        with open(saida, "w", encoding="utf-8") as f:
            f.write(conteudo)
        
        return saida
    
    except Exception as e:
        print(f"❌ Erro ao processar {entrada.name}: {e}")
        return None

def processar_pasta(
    entrada: Path,
    saida: Path,
    formato: str = "markdown",
    extensoes: set = None,
    max_workers: int = 4,
    sobreescrever: bool = False,
):
    """
    Processa todos os arquivos suportados em uma pasta recursivamente.
    
    Args:
        entrada: Diretório de entrada.
        saida: Diretório de saída.
        formato: 'markdown', 'json' ou 'text'.
        extensoes: Conjunto de extensões a processar (ex: {'.pdf', '.docx'}).
        max_workers: Número de workers paralelos.
        sobreescrever: Se True, substitui arquivos existentes.
    """
    entrada = Path(entrada)
    saida = Path(saida)
    
    if not entrada.exists() or not entrada.is_dir():
        raise ValueError(f"Diretório de entrada inválido: {entrada}")
    
    extensoes = extensoes or EXTENSOES_SUPORTADAS
    
    # Coleta todos os arquivos suportados
    arquivos = []
    for root, _, files in os.walk(entrada):
        for file in files:
            caminho = Path(root) / file
            if caminho.suffix.lower() in extensoes and caminho.is_file():
                arquivos.append(caminho)
    
    if not arquivos:
        print("⚠️ Nenhum arquivo suportado encontrado.")
        return
    
    print(f"📂 Encontrados {len(arquivos)} arquivos para processar.")
    
    # Prepara o caminho de saída preservando a estrutura
    arquivos_com_saida = []
    for arq in arquivos:
        relativo = arq.relative_to(entrada)
        destino = saida / relativo.parent
        if not sobreescrever:
            # Verifica se o arquivo já foi convertido
            if formato == "markdown":
                sufixo = ".md"
            elif formato == "json":
                sufixo = ".json"
            else:
                sufixo = ".txt"
            saida_existente = destino / (arq.stem + sufixo)
            if saida_existente.exists():
                continue
        arquivos_com_saida.append((arq, destino))
    
    if not arquivos_com_saida:
        print("✅ Todos os arquivos já foram convertidos.")
        return
    
    print(f"🔄 Processando {len(arquivos_com_saida)} arquivos...")
    
    # Processa em paralelo
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(converter_arquivo, arq, destino, formato): (arq, destino)
            for arq, destino in arquivos_com_saida
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Convertendo"):
            arq, destino = futures[future]
            resultado = future.result()
            if resultado:
                tqdm.write(f"✅ {arq.name} → {resultado}")

def main():
    parser = argparse.ArgumentParser(
        description="Processa uma pasta de documentos com Docling."
    )
    parser.add_argument(
        "entrada",
        help="Caminho da pasta com documentos a converter."
    )
    parser.add_argument(
        "--saida", "-o",
        default="./docling_output",
        help="Diretório onde salvar os arquivos convertidos (padrão: ./docling_output)."
    )
    parser.add_argument(
        "--formato", "-f",
        choices=["markdown", "json", "text"],
        default="markdown",
        help="Formato de saída (padrão: markdown)."
    )
    parser.add_argument(
        "--extensoes", "-e",
        nargs="+",
        default=[".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".html", ".txt", ".md", ".jpg", ".png"],
        help="Extensões a processar (ex: .pdf .docx)."
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Número de processos paralelos (padrão: 4)."
    )
    parser.add_argument(
        "--sobreescrever", "-s",
        action="store_true",
        help="Sobrescrever arquivos já convertidos."
    )
    
    args = parser.parse_args()
    
    # Converte extensões para set com ponto
    extensoes = set(e if e.startswith('.') else f'.{e}' for e in args.extensoes)
    
    try:
        processar_pasta(
            entrada=Path(args.entrada),
            saida=Path(args.saida),
            formato=args.formato,
            extensoes=extensoes,
            max_workers=args.workers,
            sobreescrever=args.sobreescrever,
        )
        print("🎉 Processamento concluído!")
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()