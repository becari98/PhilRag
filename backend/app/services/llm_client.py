"""
philorag.services.llm_client
─────────────────────────────
Cliente unificado para LLMs: Ollama (local) e OpenAI API.

Prioridade para o ThinkPad T420 (CPU-only):
    llama3.2:3b   → raciocínio rápido, análise simples
    mistral:7b    → análise filosófica mais robusta (mais lento)
    phi3:mini     → alternativa leve

Para embeddings via Ollama:
    nomic-embed-text  → alta qualidade, rápido em CPU
"""

from __future__ import annotations

from loguru import logger

from app.config import LLMProvider, settings


class LLMClient:
    """
    Abstração sobre provedores de LLM.
    
    Uso:
        client = LLMClient()
        response = await client.complete(messages=[{"role":"user","content":"…"}])
        embedding = await client.embed("texto")
    """

    def __init__(self, provider: LLMProvider | None = None, model: str = ""):
        self.provider = provider or settings.llm_provider
        self.model = model or settings.ollama_model

    # ── Completion ────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str:
        if self.provider == LLMProvider.OLLAMA:
            return await self._ollama_complete(messages, temperature, max_tokens)
        elif self.provider == LLMProvider.OPENAI:
            return await self._openai_complete(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Provedor não suportado: {self.provider}")

    async def _ollama_complete(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> str:
        try:
            import ollama as _ollama
        except ImportError:
            raise ImportError("ollama não instalado: pip install ollama")

        client = _ollama.AsyncClient(host=settings.ollama_base_url)
        response = await client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"]

    async def _openai_complete(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai não instalado: pip install openai")

        base_url = (
            "https://openrouter.ai/api/v1"
            if self.provider.value == "openrouter"
            else None
        )
        api_key = (
            settings.openrouter_api_key
            if self.provider.value == "openrouter"
            else settings.openai_api_key
        )

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        response = await client.chat.completions.create(
            model=self.model or settings.openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    # ── Embeddings ────────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Gera embedding para um texto."""
        if settings.embedding_provider.value == "ollama":
            return await self._ollama_embed(text)
        else:
            return await self._st_embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para múltiplos textos."""
        # Ollama não tem endpoint de batch; paralelizar com asyncio.gather
        import asyncio
        return await asyncio.gather(*[self.embed(t) for t in texts])

    async def _ollama_embed(self, text: str) -> list[float]:
        try:
            import ollama as _ollama
        except ImportError:
            raise ImportError("ollama não instalado")

        client = _ollama.AsyncClient(host=settings.ollama_base_url)
        response = await client.embeddings(
            model=settings.embedding_model,
            prompt=text,
        )
        return response["embedding"]

    async def _st_embed(self, text: str) -> list[float]:
        """Sentence Transformers (CPU local, sem Ollama)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers não instalado")

        # Carregado uma vez (singleton simples)
        if not hasattr(self, "_st_model"):
            logger.info(f"Carregando SentenceTransformer: {settings.embedding_model}")
            self._st_model = SentenceTransformer(settings.embedding_model)

        embedding = self._st_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    # ── Utilidades ────────────────────────────────────────────────────────────

    async def is_ollama_available(self) -> bool:
        """Testa conectividade com Ollama."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
                return r.status_code == 200
        except Exception:
            return False

    async def list_ollama_models(self) -> list[str]:
        """Lista modelos disponíveis no Ollama."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
