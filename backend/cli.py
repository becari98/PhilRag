#!/usr/bin/env python3
"""
PhiloRAG — Interface de Linha de Comando
─────────────────────────────────────────
Uso imediato sem frontend. Ideal para o workflow de tese.

Comandos:

    philorag index arquivo.pdf --obra ZA --autor Nietzsche --ano 1883
    philorag index comentador.pdf --obra Lemm2009

    philorag query "Reconstrua o conceito de animalidade em ZA I"
    philorag query "Compare Lemm e Cragnolini" --mode comparative
    philorag query "Qual o papel do corpo em ZA?" --obra ZA

    philorag dossier animalidade
    philorag dossier corpo --export dossiê_corpo.md

    philorag library                  # lista documentos indexados
    philorag library --status         # status completo (modelos, chunks, etc.)

Pré-requisitos:
    1. Ollama instalado e rodando (https://ollama.ai)
    2. ollama pull llama3.2:3b
    3. ollama pull nomic-embed-text
    4. pip install -r requirements.txt
    5. cp .env.example .env
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="philorag",
    help="PhiloRAG — Pesquisa filosófica com RAG hierárquico",
    rich_markup_mode="rich",
)
console = Console()


# ── Helpers assíncronos ────────────────────────────────────────────────────────

def run(coro):
    """Executa corrotina em loop de eventos."""
    return asyncio.run(coro)


def _setup():
    """Inicializa banco e diretórios."""
    sys.path.insert(0, str(Path(__file__).parent))
    from app.config import settings
    from app.database.db import init_db
    settings.ensure_dirs()
    run(init_db())
    return settings


# ── Commands ───────────────────────────────────────────────────────────────────

@app.command()
def index(
    filepath: str = typer.Argument(..., help="Caminho do arquivo a indexar"),
    obra: str = typer.Option("", "--obra", "-o", help="Sigla canônica da obra (ex.: ZA, ABM, EH)"),
    autor: str = typer.Option("", "--autor", "-a", help="Autor do texto"),
    ano: str = typer.Option("", "--ano", help="Ano de publicação"),
    idioma: str = typer.Option("pt", "--idioma", help="Idioma do texto (pt, de, en, fr)"),
    strategy: str = typer.Option("hierarchical", "--strategy", "-s",
                                  help="Estratégia de chunking: hierarchical|aphorism|token|paragraph"),
    summaries: bool = typer.Option(True, "--summaries/--no-summaries",
                                    help="Gerar sumários por LLM"),
    concepts: bool = typer.Option(True, "--concepts/--no-concepts",
                                   help="Gerar mapa conceitual por LLM"),
):
    """
    Indexa um documento no corpus filosófico.

    [bold]Exemplos:[/bold]
        philorag index ZA.pdf --obra ZA --autor Nietzsche --ano 1883 --strategy aphorism
        philorag index lemm_nietzsche_animal.pdf --obra Lemm2009 --autor Lemm
    """
    _setup()
    from app.database.db import async_session
    from app.models.document import ChunkStrategy, DocumentCreate
    from app.services.indexer import IndexingService

    strat_map = {
        "hierarchical": ChunkStrategy.HIERARCHICAL,
        "aphorism":     ChunkStrategy.APHORISM,
        "token":        ChunkStrategy.TOKEN,
        "paragraph":    ChunkStrategy.PARAGRAPH,
    }
    chunk_strat = strat_map.get(strategy, ChunkStrategy.HIERARCHICAL)

    config = DocumentCreate(
        obra=obra,
        autor=autor,
        ano=ano,
        idioma=idioma,
        chunk_strategy=chunk_strat,
        generate_summaries=summaries,
        generate_concept_map=concepts,
    )

    async def _run():
        async with async_session() as session:
            svc = IndexingService()
            with console.status(f"[cyan]Indexando {filepath}…[/cyan]"):
                doc_id = await svc.index(filepath, config, session)
            return doc_id

    doc_id = run(_run())
    rprint(Panel(
        f"[green]✓ Indexação concluída[/green]\n"
        f"document_id: [bold]{doc_id}[/bold]\n"
        f"Obra: {obra or '(auto-detectada)'}  |  Autor: {autor or '(auto-detectado)'}",
        title="PhiloRAG — Index",
    ))


@app.command()
def query(
    question: str = typer.Argument(..., help="Pergunta filosófica"),
    mode: str = typer.Option("free", "--mode", "-m",
                              help="Modo: exegetical|bibliographic|comparative|dossier|free"),
    obra: str = typer.Option("", "--obra", help="Filtrar por obra (ex.: ZA)"),
    autor: str = typer.Option("", "--autor", help="Filtrar por autor"),
    top_k: int = typer.Option(8, "--top-k", help="Número de chunks recuperados"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="Exibir fontes"),
):
    """
    Consulta o corpus filosófico com RAG hierárquico.

    [bold]Exemplos:[/bold]
        philorag query "Reconstrua o conceito de animalidade em ZA I" --mode exegetical --obra ZA
        philorag query "Quais comentadores discutem a crítica ao antropocentrismo?" --mode bibliographic
        philorag query "Compare Lemm e Cragnolini sobre a animalidade" --mode comparative
    """
    _setup()
    from app.database.db import async_session, DocumentORM
    from app.models.query import QueryMode, QueryRequest
    from app.services.rag_pipeline import HierarchicalRAGPipeline
    from app.services.vector_store import VectorStore
    from app.services.llm_client import LLMClient
    from sqlalchemy import select
    import json as _json

    mode_map = {
        "exegetical":    QueryMode.EXEGETICAL,
        "bibliographic": QueryMode.BIBLIOGRAPHIC,
        "comparative":   QueryMode.COMPARATIVE,
        "dossier":       QueryMode.DOSSIER,
        "free":          QueryMode.FREE,
    }
    qmode = mode_map.get(mode, QueryMode.FREE)

    req = QueryRequest(
        question=question,
        mode=qmode,
        obra_sigla=obra,
        autor=autor,
        top_k=top_k,
    )

    async def _run():
        async with async_session() as session:
            result = await session.execute(select(DocumentORM))
            docs = result.scalars().all()
            doc_summaries = {d.id: d.summary_work for d in docs if d.summary_work}
            concept_maps = {
                d.obra: _json.loads(d.concept_map or "[]")
                for d in docs if d.obra and d.concept_map
            }
            pipeline = HierarchicalRAGPipeline()
            return await pipeline.query(req, doc_summaries, {}, concept_maps)

    with console.status("[cyan]Consultando corpus filosófico…[/cyan]"):
        response = run(_run())

    console.print()
    console.print(Panel(
        Markdown(response.answer),
        title=f"[bold blue]PhiloRAG — {qmode.value.upper()}[/bold blue]",
        subtitle=f"[dim]{response.chunks_retrieved} chunks | ~{response.tokens_in_context} tokens | {response.latency_ms}ms[/dim]",
    ))

    if show_sources and response.citations:
        console.print()
        table = Table(title="Fontes recuperadas", show_lines=True)
        table.add_column("N°", style="dim", width=4)
        table.add_column("Obra", style="cyan", width=15)
        table.add_column("Capítulo", width=25)
        table.add_column("Score", style="green", width=6)
        table.add_column("Trecho", width=60)
        for i, c in enumerate(response.citations, 1):
            table.add_row(
                str(i),
                c.metadata_obra or "—",
                c.title[:25] or "—",
                f"{c.score:.2f}",
                c.text[:100] + "…" if len(c.text) > 100 else c.text,
            )
        console.print(table)


@app.command()
def dossier(
    theme: str = typer.Argument(..., help="Tema do dossiê (ex.: animalidade, corpo, além-do-homem)"),
    export: str = typer.Option("", "--export", "-e", help="Exportar para arquivo .md"),
    citation_style: str = typer.Option("abnt", "--style", help="abnt|chicago|apa"),
):
    """
    Gera um dossiê temático completo.

    [bold]Exemplo:[/bold]
        philorag dossier animalidade --export dossie_animalidade.md
        philorag dossier "crítica ao antropocentrismo" --style abnt
    """
    _setup()
    from app.database.db import async_session, DocumentORM
    from app.models.query import DossierRequest, QueryMode, QueryRequest
    from app.services.rag_pipeline import HierarchicalRAGPipeline
    from sqlalchemy import select
    import json as _json

    async def _run():
        async with async_session() as session:
            result = await session.execute(select(DocumentORM))
            docs = result.scalars().all()
            doc_summaries = {d.id: d.summary_work for d in docs if d.summary_work}
            concept_maps = {
                d.obra: _json.loads(d.concept_map or "[]")
                for d in docs if d.obra and d.concept_map
            }
            pipeline = HierarchicalRAGPipeline()

            sections = []
            titles = [
                ("Delimitação conceitual", f"Delimite filosoficamente o conceito de '{theme}'"),
                ("Textos primários", f"Trace o conceito de '{theme}' nos textos primários"),
                ("Recepção crítica", f"Como os comentadores tratam '{theme}'?"),
                ("Debates em aberto", f"Quais os debates interpretativos sobre '{theme}'?"),
            ]
            modes = [
                QueryMode.EXEGETICAL, QueryMode.EXEGETICAL,
                QueryMode.BIBLIOGRAPHIC, QueryMode.COMPARATIVE,
            ]

            for (title, q), m in zip(titles, modes):
                r = await pipeline.query(
                    QueryRequest(question=q, mode=m),
                    doc_summaries, {}, concept_maps,
                )
                sections.append((title, r.answer, r.citations))

            return sections

    with console.status(f"[cyan]Gerando dossiê: {theme}…[/cyan]"):
        sections = run(_run())

    # Monta Markdown
    md_parts = [f"# Dossiê: {theme}\n"]
    for i, (title, content, cits) in enumerate(sections, 1):
        md_parts.append(f"\n## {i}. {title}\n\n{content}\n")
        if cits:
            md_parts.append("\n**Fontes:**")
            for c in cits[:5]:
                obra = c.metadata_obra or "?"
                md_parts.append(f"- [{c.score:.2f}] {obra} — {c.title[:50]}")

    md_content = "\n".join(md_parts)

    if export:
        Path(export).write_text(md_content, encoding="utf-8")
        rprint(f"[green]✓ Dossiê exportado: {export}[/green]")
    else:
        console.print(Markdown(md_content))


@app.command()
def library(
    status: bool = typer.Option(False, "--status", help="Exibir status completo"),
):
    """Lista os documentos indexados na biblioteca."""
    _setup()
    from app.database.db import async_session, DocumentORM
    from sqlalchemy import select

    async def _run():
        async with async_session() as session:
            result = await session.execute(
                select(DocumentORM).order_by(DocumentORM.created_at.desc())
            )
            return result.scalars().all()

    docs = run(_run())

    if not docs:
        rprint("[yellow]Biblioteca vazia. Use [bold]philorag index arquivo.pdf[/bold] para indexar.[/yellow]")
        return

    table = Table(title="PhiloRAG — Biblioteca Filosófica", show_lines=True)
    table.add_column("Obra", style="bold cyan", width=12)
    table.add_column("Autor", width=20)
    table.add_column("Arquivo", width=30)
    table.add_column("Chunks", style="green", width=8)
    table.add_column("Status", width=10)
    table.add_column("Indexado em", style="dim", width=20)

    for d in docs:
        status_color = "green" if d.status == "indexed" else "yellow"
        table.add_row(
            d.obra or "—",
            d.autor or "—",
            d.filename[:30],
            str(d.chunk_count),
            f"[{status_color}]{d.status}[/{status_color}]",
            d.created_at.strftime("%d/%m/%Y %H:%M"),
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(docs)} documentos[/dim]")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
