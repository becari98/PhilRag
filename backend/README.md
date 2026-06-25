# 📘 PhiloRAG – Sistema de Pesquisa Filosófica com RAG Hierárquico

**PhiloRAG** é um sistema desktop/API para pesquisa acadêmica avançada em Filosofia, especialmente voltado para teses, dissertações e trabalho exegético. Ele transforma uma biblioteca de livros, artigos e documentos em uma base de conhecimento semântica, permitindo consultas complexas a partir de LLMs locais (Ollama) ou APIs (OpenAI, OpenRouter) sem perder o contexto global das obras.

---

## 🧠 O que o PhiloRAG faz?

- **Indexação inteligente** de PDFs, DOCX, EPUB, TXT, MD, HTML e imagens (com OCR opcional).
- **Chunking hierárquico** (obra → capítulo → parágrafo → sentença) que preserva a estrutura argumentativa.
- **Geração automática** de sumários, mapas conceituais e metadados (título, autor, ano, conceitos‑chave).
- **Busca semântica** sobre os textos usando embeddings (via Ollama ou Sentence‑Transformers) e banco vetorial (ChromaDB).
- **RAG hierárquico**: ao fazer uma pergunta, o sistema injeta no LLM o sumário da obra, o capítulo relevante, os parágrafos mais similares e o mapa conceitual – tudo para respostas profundas e contextualizadas.
- **Modos de consulta**: exegética, bibliográfica, comparativa, dossiê temático e livre.
- **Geração de dossiês** completos sobre um conceito (ex: "animalidade") com textos primários, comentadores, debates e bibliografia.
- **Exportação** para Markdown, JSON, BibTeX e GraphML.
- **Interface API** REST (FastAPI) com documentação interativa via Swagger.

---

## ⚙️ Tecnologias utilizadas

- **Backend**: Python 3.12 + FastAPI + Uvicorn
- **Banco de dados**: SQLite (metadados) + ChromaDB (vetores)
- **LLMs**: Ollama (local) ou OpenAI/OpenRouter (nuvem)
- **Embeddings**: Ollama (nomic‑embed‑text, all‑minilm) ou Sentence‑Transformers (all‑MiniLM‑L6‑v2)
- **Parsing de documentos**: PyMuPDF, python‑docx, ebooklib, html2text, BeautifulSoup (fallback) – com suporte a Docling (opcional)
- **OCR**: Tesseract (fallback) ou Docling (integrado)

---

## 📦 Requisitos de sistema

- **Sistema operacional**: Linux (Ubuntu/Debian) ou Windows 10/11 (WSL2 recomendado) ou macOS
- **Python** 3.12 ou superior
- **Memória RAM**: mínimo 8 GB (16 GB recomendado para modelos LLM 3B–7B)
- **Espaço em disco**: pelo menos 10 GB para documentos e banco vetorial
- **Ollama** (para LLMs locais) – [instalação](https://ollama.com/download)

---

## 🚀 Instalação

### 1. Clone ou copie o projeto
```bash
git clone https://github.com/seu-usuario/philorag.git
cd philorag/backend
```
*(Se você já tem os arquivos, apenas entre na pasta `backend`)*

### 2. Crie um ambiente virtual

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (cmd):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instale as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Atenção**: Para garantir que o PyTorch seja instalado na versão CPU (sem CUDA) – essencial para quem não tem placa NVIDIA – faça **antes**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Depois instale o restante.

### 4. Instale o Ollama (para LLMs locais)

Siga a [documentação oficial](https://ollama.com/download) para seu sistema. Após instalar, inicie o serviço:
```bash
ollama serve   # (em segundo plano)
```

Baixe pelo menos um modelo de chat e um de embedding:
```bash
ollama pull llama3.2:3b          # modelo de chat leve
ollama pull all-minilm           # modelo de embedding leve
```

### 5. Configure o arquivo `.env`

Crie um arquivo `.env` na raiz do `backend` com o seguinte conteúdo (ajuste conforme sua necessidade):

```ini
# LLM
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434

# Embeddings
EMBEDDING_PROVIDER=ollama          # ou sentence_transformers
EMBEDDING_MODEL=all-minilm         # ou nomic-embed-text, etc.
EMBEDDING_DIMENSION=384            # 768 para nomic-embed-text, 384 para all-minilm

# Banco de dados
DATABASE_URL=sqlite+aiosqlite:///./data/philorag.db
VECTOR_STORE_PATH=./data/chromadb
CHROMA_COLLECTION_NAME=philorag

# Diretórios de documentos
DOCUMENTS_PATH=./data/documents
PROCESSED_PATH=./data/processed

# Chunking
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=64

# RAG
RAG_TOP_K=8
RAG_RERANK=true

# Opcional – chaves para OpenAI/OpenRouter
# OPENAI_API_KEY=sk-...
# OPENROUTER_API_KEY=...
```

---

## ▶️ Execução

### Iniciar o servidor API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- `--reload`: recarrega automaticamente ao alterar o código (útil para desenvolvimento).
- `--host 0.0.0.0`: permite acesso externo (se quiser apenas local, use `127.0.0.1`).
- `--port 8000`: porta padrão.

Acesse a documentação interativa: [http://localhost:8000/docs](http://localhost:8000/docs)

### Usar a CLI (opcional)

Se o arquivo `cli.py` estiver implementado, você pode executar comandos diretamente:

```bash
python cli.py import --path /caminho/para/documento.pdf
python cli.py ask "O que Nietzsche diz sobre a vontade de potência?"
python cli.py list
```

---

## 📄 Indexar um documento

Via **Swagger UI**:
1. Acesse `/docs`
2. Endpoint `POST /documents/`
3. Envie o arquivo e preencha os metadados (obra, autor, ano, etc.).

Via **curl**:
```bash
curl -X POST "http://localhost:8000/documents/" \
  -F "file=@/caminho/para/ecce_homo.pdf" \
  -F "obra=EH" \
  -F "autor=Nietzsche" \
  -F "ano=1888" \
  -F "chunk_strategy=hierarchical"
```

---

## 🔍 Fazer uma pergunta

Via **Swagger UI**: endpoint `POST /query`

Via **curl**:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Como Nietzsche concebe a relação entre corpo e alma?",
    "mode": "exegetical",
    "top_k": 6
  }'
```

---

## 🧩 Estrutura de diretórios do projeto

```
backend/
├── app/
│   ├── config.py
│   ├── database/           # SQLAlchemy models e init_db
│   ├── models/             # Pydantic schemas
│   ├── routers/            # Endpoints FastAPI
│   └── services/           # Lógica de negócio (indexador, chunker, RAG, LLM, vector store)
├── data/                   # (criado automaticamente)
│   ├── documents/          # arquivos originais
│   ├── processed/          # arquivos processados (markdown)
│   ├── chromadb/           # banco vetorial
│   └── philorag.db         # banco SQLite
├── main.py                 # entry point
├── cli.py                  # interface de linha de comando (opcional)
├── requirements.txt
├── .env                    # configurações
└── venv/                   # ambiente virtual
```

---

## 🔧 Dicas para Windows

- Use **WSL2** (Ubuntu) para uma experiência idêntica ao Linux.
- Se usar PowerShell/CMD nativo, substitua barras `\` por `/` nos caminhos (ex: `C:/Users/...`).
- O Ollama para Windows está disponível e funciona da mesma forma.
- Para ativar o venv: `venv\Scripts\activate` (CMD) ou `.\venv\Scripts\Activate.ps1` (PowerShell).

---

## ⚖️ PhiloRAG vs. Docling vs. MarkItDown

| Ferramenta    | Objetivo principal | Tipo de saída | OCR | Estrutura hierárquica | RAG integrado |
|---------------|---------------------|---------------|-----|------------------------|---------------|
| **PhiloRAG**  | Pesquisa filosófica com RAG | API, Markdown, JSON | Opcional (Tesseract/Docling) | ✅ Sim (obra→capítulo→parágrafo) | ✅ Completo |
| **Docling**   | Conversão de documentos (IBM) | Markdown, JSON, CSV | ✅ Sim (nativo) | ⚠️ Detecta headings, mas não gera estrutura semântica | ❌ Apenas conversão |
| **MarkItDown**| Conversão de documentos (Microsoft) | Markdown, JSON | ⚠️ Limitado (via OCR externo) | ❌ Não preserva hierarquia | ❌ Apenas conversão |

**Resumo**:  
- Use **PhiloRAG** se quiser **pesquisar** e **analisar** filosoficamente um corpus, com LLMs e contexto hierárquico.  
- Use **Docling** ou **MarkItDown** se precisar apenas **extrair texto** de documentos para outros fins (ex: preparar dados para outro sistema).  
- PhiloRAG pode usar Docling como *backend de extração* (opcional) para melhorar a qualidade do parsing, mas já possui seu próprio processador legado.

---

## 📚 Exemplo de fluxo de trabalho

1. Indexe uma obra:
   ```bash
   python cli.py import --path "Nietzsche_ZA.pdf" --obra "ZA" --autor "Nietzsche"
   ```
2. Faça uma pergunta exegética:
   ```bash
   python cli.py ask "Explique o conceito de Übermensch em Assim Falou Zaratustra"
   ```
3. Gere um dossiê temático:
   ```bash
   python cli.py dossier --tema "vontade de potência" --format md
   ```
4. Consulte via API ou frontend (se disponível).

---

## 🤝 Contribuição

Sinta‑se à vontade para abrir issues ou enviar pull requests. O projeto é aberto e visa auxiliar pesquisadores em humanidades.

---

## 📄 Licença

Este projeto está sob a licença MIT – veja o arquivo `LICENSE` para detalhes.

---

**Desenvolvido com ❤️ para a comunidade filosófica.**
