# 📄 Docling Tools – Conversor de Documentos

Este diretório contém scripts para converter documentos (PDF, DOCX, imagens, etc.) para Markdown, JSON ou texto usando o **Docling** da IBM.

---

## 📋 Pré‑requisitos

- Python 3.12 (já instalado no sistema)
- Ambiente virtual `docling_env` com todas as dependências
- Arquivos a converter (PDF, DOCX, PPTX, XLSX, imagens, etc.)

---

## 🚀 Ativar o ambiente virtual

Sempre que for usar os scripts, ative o venv:

```bash
source ~/Área\ de\ trabalho/philorag/docling/docling_env/bin/activate
```

Para desativar:
```bash
deactivate
```

---

## 📂 Estrutura de diretórios

```
docling/
├── docling_env/          # Ambiente virtual com Docling
├── docling_convert.py    # Converte um único arquivo
├── docling_batch.py      # Converte uma pasta inteira (recursivo)
└── README.md             # Este arquivo
```

---

## 🧪 1. Converter um único arquivo

**Script:** `docling_convert.py`

```bash
python docling_convert.py <caminho_do_arquivo>
```

### Exemplo
```bash
python docling_convert.py ~/Documentos/artigo.pdf
```

**Saída:** Gera um arquivo `.md` no mesmo diretório com o mesmo nome base.

### Opções extras (se usar a versão com argumentos)
```bash
python docling_convert.py arquivo.pdf --formato json --saida saida.json
```

---

## 📁 2. Converter uma pasta inteira (em lote)

**Script:** `docling_batch.py`

```bash
python docling_batch.py <pasta_entrada> [--saida PASTA_SAIDA] [--formato md|json|text] [--extensoes .pdf .docx ...] [--workers N] [--sobreescrever]
```

### Exemplo básico
```bash
python docling_batch.py ~/Documentos/Teses
```

Isso processa todos os arquivos suportados dentro de `Teses` (e subpastas) e salva os Markdown em `./docling_output`.

### Exemplo com opções
```bash
# Converter apenas PDFs e DOCXs, salvar em ./output, usar 6 workers
python docling_batch.py ~/Documentos --saida ./output --extensoes .pdf .docx --workers 6

# Converter para JSON e sobrescrever arquivos existentes
python docling_batch.py ~/Documentos --formato json --sobreescrever
```

---

## ⚙️ Parâmetros do `docling_batch.py`

| Parâmetro          | Descrição |
|--------------------|-----------|
| `entrada`          | **(obrigatório)** Pasta com os documentos |
| `--saida`, `-o`    | Pasta onde salvar os arquivos convertidos (padrão: `./docling_output`) |
| `--formato`, `-f`  | `markdown` (padrão), `json` ou `text` |
| `--extensoes`, `-e`| Extensões a processar (ex: `.pdf .docx .jpg`). Padrão: todas suportadas |
| `--workers`, `-w`  | Número de processos paralelos (padrão: 4). Para T420, use `4` ou `6`. |
| `--sobreescrever`, `-s` | Se presente, substitui arquivos já convertidos |

---

## 💡 Dicas para o ThinkPad T420

- Use `--workers 4` para não sobrecarregar a CPU (o processador tem 4 núcleos físicos).
- Para PDFs grandes, aumente o timeout padrão do Docling (não é necessário nos scripts, mas se travar, considere processar em lotes menores).
- Monitore a temperatura com `sensors` se fizer conversões longas.

---

## 🧹 Limpeza

Para remover os arquivos gerados:
```bash
rm -rf ./docling_output   # ou a pasta que você definiu em --saida
```

---

## ❓ Perguntas frequentes

**Posso converter arquivos de imagem?**  
Sim, o Docling suporta JPG, PNG, BMP, TIFF.

**O Docling faz OCR?**  
Sim, se instalado com os extras (`pip install docling[easyocr]`). O ambiente atual já inclui suporte básico.

**O que acontece se o arquivo já foi convertido?**  
Por padrão, o batch pula arquivos já existentes. Use `--sobreescrever` para forçar a recriação.

---

## 📦 Dependências (já instaladas no venv)

- `docling`
- `torch`, `torchvision` (versão CPU)
- `tqdm` (barra de progresso)
- `transformers`, `huggingface-hub`

---

## 🧑‍💻 Suporte

Para dúvidas, consulte a [documentação oficial do Docling](https://docling-project.github.io/docling/).
