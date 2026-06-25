"""
philorag.services.rag_pipeline
────────────────────────────────
Pipeline RAG hierárquico para pesquisa filosófica.

A inovação central do PhiloRAG é a injeção de contexto em múltiplos níveis:

    [1] Sumário da obra  →  situa a pergunta no projeto global do autor
    [2] Sumários dos capítulos relevantes  →  contexto argumentativo imediato
    [3] Chunks semânticos (parágrafo)  →  passagens textuais diretas
    [4] Mapa conceitual  →  relações entre conceitos recuperados
    [5] Metadados  →  referências bibliográficas precisas

Modos de consulta com prompts específicos:
    EXEGETICAL    →  análise imanente, rigor conceitual
    BIBLIOGRAPHIC →  rastreamento de comentadores
    COMPARATIVE   →  contraste entre intérpretes
    DOSSIER       →  síntese temática completa
    FREE          →  consulta aberta
"""

from __future__ import annotations

import time
from loguru import logger

from app.models.query import (
    ChunkCitation, QueryMode, QueryRequest, RAGResponse,
)
from app.services.llm_client import LLMClient
from app.services.vector_store import SearchResult, VectorStore


# ── Prompts por modo ──────────────────────────────────────────────────────────

SYSTEM_PROMPTS: dict[QueryMode, str] = {
    QueryMode.EXEGETICAL: """Você é um especialista em filosofia continental e exegese textual.
Sua tarefa é realizar uma análise filosófica rigorosa com base exclusivamente nos textos fornecidos.

Princípios metodológicos:
- Analise os textos de forma imanente, seguindo a argumentação do próprio autor
- Precise os termos técnicos em seu idioma original quando relevante
- Identifique distinções conceituais sutis
- Sinalize tensões ou desenvolvimentos no corpus
- Cite com precisão: [Obra, §/cap., p.] quando possível
- Evite projeções externas ao corpus fornecido
- Responda em português acadêmico rigoroso""",

    QueryMode.BIBLIOGRAPHIC: """Você é um especialista em bibliografia filosófica especializada.
Identifique e mapeie como os comentadores discutem o tema proposto.

Estruture a resposta:
1. Quais comentadores abordam o tema
2. Perspectivas interpretativas distintas
3. Pontos de convergência e divergência
4. Lacunas bibliográficas identificáveis
Cite autores e obras com precisão (ABNT).""",

    QueryMode.COMPARATIVE: """Você é um especialista em hermenêutica filosófica comparada.
Compare rigorosamente as posições dos intérpretes presentes no corpus.

Estruture:
1. Pontos de partida interpretativos de cada autor
2. Divergências fundamentais (metodológicas, conceituais)
3. Posições medianas ou sínteses possíveis
4. Avaliação crítica das interpretações
Seja preciso quanto às fontes primárias mobilizadas por cada intérprete.""",

    QueryMode.DOSSIER: """Você é um pesquisador filosófico sintetizando um dossiê temático completo.
Organize o material de forma sistemática e exaustiva.

Estruture o dossiê em:
1. Delimitação do conceito / problema
2. Tratamento nos textos primários (cronológico ou temático)
3. Recepção crítica (principais intérpretes)
4. Debates em aberto
5. Referências bibliográficas (ABNT)
O dossiê deve ser utilizável como base para capítulo de tese.""",

    QueryMode.FREE: """Você é um assistente especializado em filosofia acadêmica.
Responda com rigor conceitual, base nos textos fornecidos e precisão bibliográfica.
Prefira o português acadêmico. Cite as fontes quando relevante.""",
}

# ── Pipeline ──────────────────────────────────────────────────────────────────

class HierarchicalRAGPipeline:
    """
    Pipeline RAG com injeção de contexto hierárquico.
    
    Fluxo:
        1. Embed da pergunta
        2. Busca vetorial (top-k chunks parágrafo)
        3. Expansão: recupera sumários dos capítulos pai
        4. Recupera sumário(s) de obra
        5. Recupera mapa conceitual
        6. Monta prompt estruturado
        7. LLM → resposta
        8. Formata resposta com citações
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.vs = vector_store or VectorStore()
        self.llm = llm_client or LLMClient()

    async def query(
        self,
        request: QueryRequest,
        document_summaries: dict[str, str] | None = None,
        chunk_parents: dict[str, dict] | None = None,
        concept_maps: dict[str, list[dict]] | None = None,
    ) -> RAGResponse:
        """
        Executa o pipeline RAG completo.
        
        Args:
            request: QueryRequest com pergunta e parâmetros
            document_summaries: {document_id: summary_work}
            chunk_parents: {chunk_id: {title, summary, obra, ...}}
            concept_maps: {obra: [{conceito, relacionado_com}, ...]}
        """
        t0 = time.time()

        # 1. Embed da pergunta
        query_embedding = await self.llm.embed(request.question)

        # 2. Busca semântica
        raw_results = self.vs.search(
            query_embedding=query_embedding,
            top_k=request.top_k * 2,            # busca mais ampla antes do rerank
            document_ids=[str(d) for d in request.document_ids] or None,
            obra=request.obra_sigla,
            autor=request.autor,
        )

        if not raw_results:
            return RAGResponse(
                question=request.question,
                mode=request.mode,
                answer="Nenhum chunk relevante encontrado no corpus. "
                       "Verifique se os documentos foram indexados.",
                chunks_retrieved=0,
            )

        # 3. Reranking
        if len(raw_results) > request.top_k:
            raw_results = VectorStore.rerank(request.question, raw_results, request.top_k)
        else:
            raw_results = raw_results[:request.top_k]

        # 4. Monta contexto hierárquico
        context_parts: list[str] = []
        work_summaries_used: list[str] = []
        chapter_summaries_used: list[str] = []
        concept_map_used: list[str] = []

        # 4a. Sumários de obra (level 1)
        if request.include_summaries and document_summaries:
            seen_docs = {r.document_id for r in raw_results}
            for doc_id in seen_docs:
                summary = document_summaries.get(doc_id, "")
                if summary:
                    obra = next(
                        (r.obra for r in raw_results if r.document_id == doc_id), ""
                    )
                    label = f"[SUMÁRIO DA OBRA: {obra or doc_id}]"
                    context_parts.append(f"{label}\n{summary}")
                    work_summaries_used.append(label)

        # 4b. Sumários de capítulo (level 2) — expandidos a partir dos chunks pai
        if request.include_summaries and chunk_parents:
            seen_chapters = set()
            for r in raw_results:
                parent = chunk_parents.get(r.chunk_id)
                if parent:
                    chapter_key = f"{r.obra}::{parent.get('title', '')}"
                    if chapter_key not in seen_chapters:
                        seen_chapters.add(chapter_key)
                        chapter_summary = parent.get("summary", "")
                        if chapter_summary:
                            label = f"[SUMÁRIO DO CAPÍTULO: {parent.get('title', '')} — {r.obra}]"
                            context_parts.append(f"{label}\n{chapter_summary}")
                            chapter_summaries_used.append(label)

        # 4c. Mapa conceitual relevante
        if request.include_concept_map and concept_maps:
            obras_in_results = {r.obra for r in raw_results}
            query_terms = set(request.question.lower().split())
            for obra, cmap in concept_maps.items():
                if obra not in obras_in_results:
                    continue
                for rel in cmap:
                    concept = rel.get("conceito", "").lower()
                    if any(t in concept or concept in t for t in query_terms):
                        yaml_line = (
                            f"conceito: {rel['conceito']}\n"
                            f"  relacionado_com: {', '.join(rel.get('relacionado_com', []))}"
                        )
                        concept_map_used.append(yaml_line)

            if concept_map_used:
                context_parts.append(
                    "[MAPA CONCEITUAL RELEVANTE]\n" + "\n".join(concept_map_used)
                )

        # 4d. Chunks principais (parágrafos — level 3)
        context_parts.append("[PASSAGENS TEXTUAIS RELEVANTES]")
        for i, r in enumerate(raw_results, 1):
            ref = f"{r.obra} — {r.capitulo or r.title}" if (r.obra or r.capitulo or r.title) else f"Chunk {i}"
            author_info = f" ({r.autor})" if r.autor else ""
            context_parts.append(f"[{i}] {ref}{author_info}\n{r.text}")

        context = "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(context_parts)

        # 5. Monta prompt
        system_prompt = SYSTEM_PROMPTS[request.mode]
        user_message = self._build_user_prompt(request, context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]

        # 6. LLM
        logger.info(
            f"RAG [{request.mode.value}]: {len(raw_results)} chunks, "
            f"~{len(context)//4} tokens de contexto"
        )
        answer = await self.llm.complete(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # 7. Formata citações
        citations = [
            ChunkCitation(
                chunk_id=r.chunk_id,       # type: ignore[arg-type]
                document_id=r.document_id,  # type: ignore[arg-type]
                score=r.score,
                level=r.level,
                title=r.title,
                text=r.text[:400] + "…" if len(r.text) > 400 else r.text,
                metadata_obra=r.obra,
                metadata_autor=r.autor,
                metadata_ano="",
            )
            for r in raw_results
        ]

        latency = int((time.time() - t0) * 1000)

        return RAGResponse(
            question=request.question,
            mode=request.mode,
            answer=answer,
            citations=citations,
            work_summaries_used=work_summaries_used,
            chapter_summaries_used=chapter_summaries_used,
            concept_map_used=concept_map_used,
            chunks_retrieved=len(raw_results),
            tokens_in_context=len(context) // 4,
            model_used=self.llm.model,
            latency_ms=latency,
        )

    # ── Geração de sumários ───────────────────────────────────────────────────

    async def generate_summary(self, text: str, level: str = "capítulo") -> str:
        """Gera sumário filosófico de um trecho."""
        prompt = f"""Elabore um sumário filosófico conciso (máx. 200 palavras) do seguinte {level}.
Identifique: tese central, conceitos-chave, movimento argumentativo.
Responda em português.

TEXTO:
{text[:4000]}"""

        messages = [{"role": "user", "content": prompt}]
        return await self.llm.complete(messages, temperature=0.2, max_tokens=400)

    async def generate_concept_map(self, text: str, obra: str = "") -> list[dict]:
        """Extrai mapa conceitual de um texto."""
        prompt = f"""Extraia os conceitos filosóficos centrais do seguinte texto de {obra or 'filosofia'}.
Para cada conceito, liste os termos relacionados presentes no texto.

Responda APENAS em JSON com o formato:
[
  {{"conceito": "animalidade", "relacionado_com": ["corpo", "instinto", "vida"]}},
  ...
]

TEXTO:
{text[:3000]}"""

        messages = [{"role": "user", "content": prompt}]
        try:
            import json
            raw = await self.llm.complete(messages, temperature=0.1, max_tokens=800)
            # Extrai JSON do response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except Exception as e:
            logger.warning(f"Falha ao extrair mapa conceitual: {e}")
        return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(request: QueryRequest, context: str) -> str:
        mode_instructions = {
            QueryMode.EXEGETICAL:    "Realize uma análise exegética rigorosa:",
            QueryMode.BIBLIOGRAPHIC: "Mapeie os comentadores e referências bibliográficas relevantes:",
            QueryMode.COMPARATIVE:   "Compare as posições interpretativas presentes no corpus:",
            QueryMode.DOSSIER:       "Elabore um dossiê temático completo sobre:",
            QueryMode.FREE:          "Responda com rigor filosófico:",
        }
        instruction = mode_instructions.get(request.mode, "Responda:")

        return f"""{instruction}

{request.question}

─────────────────────────────────────────────────────────────────
CORPUS FILOSÓFICO DISPONÍVEL:
{context}
─────────────────────────────────────────────────────────────────

Baseie sua resposta exclusivamente nos textos acima.
Cite as fontes com precisão (número entre colchetes [1], [2], etc. indicando a passagem correspondente).
"""
