"""
philorag — FastAPI Application
───────────────────────────────
Ponto de entrada do servidor.

Inicia o banco, monta os routers e expõe a API REST.
Documentação interativa: http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.database.db import init_db
from app.routers import documents, queries


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialização e finalização do ciclo de vida da app."""
    logger.info("═" * 60)
    logger.info("  PhiloRAG — Sistema de pesquisa filosófica com RAG")
    logger.info("═" * 60)
    settings.ensure_dirs()
    await init_db()
    logger.info(f"Banco de dados: {settings.database_url}")
    logger.info(f"Vector store  : {settings.vector_store_path}")
    logger.info(f"LLM           : {settings.llm_provider.value} / {settings.ollama_model}")
    logger.info(f"Embeddings    : {settings.embedding_model}")
    logger.info("Servidor pronto. Acesse /docs para a documentação interativa.")
    yield
    logger.info("PhiloRAG encerrado.")


app = FastAPI(
    title="PhiloRAG",
    description=(
        "Sistema de pesquisa filosófica com RAG hierárquico.\n\n"
        "Permite consultas exegéticas, bibliográficas e temáticas "
        "a um corpus filosófico indexado localmente."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (para frontend React local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(documents.router)
app.include_router(queries.router)


@app.get("/")
async def root():
    return {
        "name": "PhiloRAG",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "llm_provider": settings.llm_provider.value}
