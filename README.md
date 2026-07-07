<p align="center">
<a href="https://www.linkedin.com/in/soriamaximilianorodrigo/" target="_blank" rel="noopener noreferrer">
<img width="100%" height="100%" src="docs/img/banner.gif" alt="rag-pipeline-eval"></a>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/RAG-LangChain_%7C_LlamaIndex-1C3C3C" alt="RAG"></a>
  <a href="#"><img src="https://img.shields.io/badge/Vector_Store-Chroma_%7C_Qdrant-FF4B4B" alt="Vector Store"></a>
  <a href="#"><img src="https://img.shields.io/badge/Eval-RAGAS-0F6E56" alt="Eval"></a>
  <a href="#"><img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
</p>

<p align="center">
  <a href="https://github.com/DietrichGebert/ponytail"><img src="https://img.shields.io/badge/built_with-ponytail-111111?style=flat-square" alt="ponytail"></a>
  <img src="https://img.shields.io/badge/layout-src%2Fpackage-2088FF?style=flat-square" alt="src layout">
  <img src="https://img.shields.io/badge/license-MIT-success?style=flat-square" alt="MIT">
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=7C3AED&center=true&vCenter=true&width=820&lines=Retrieval-Augmented+Generation%2C+medido+no+asumido;faithfulness+%C2%B7+relevancia+%C2%B7+context+precision%2Frecall;LlamaIndex+%2B+Chroma+%2B+Claude+%2B+RAGAS" alt="typing SVG">
</p>

---

# rag-pipeline-eval

Pipeline de **Retrieval-Augmented Generation** sobre un corpus propio, con un módulo de **evaluación automatizada** que mide la calidad de las respuestas (faithfulness y relevancia) en lugar de asumirla.

<details>
<summary><b>📑 Tabla de contenidos</b></summary>

- [Objetivo](#objetivo)
- [Stack tecnológico sugerido](#stack-tecnológico-sugerido)
- [Estructura de carpetas propuesta](#estructura-de-carpetas-propuesta)
- [Puesta en marcha (scaffold generado)](#puesta-en-marcha-scaffold-generado)

</details>

## Objetivo

Construir un RAG end-to-end, reproducible y medible: ingerir documentos propios, trocearlos (chunking), generar embeddings, almacenarlos en un vector store, recuperar contexto relevante y generar respuestas fundamentadas. El diferencial del proyecto no es "que responda", sino **demostrar con métricas** que responde bien: sin alucinar (faithfulness), atendiendo a la pregunta (answer relevancy) y recuperando el contexto correcto (context precision/recall).

El resultado es un servicio consultable (API o CLI) más un reporte de evaluación que se puede regenerar en cada cambio del pipeline, para tomar decisiones de diseño con datos.

## Stack tecnológico sugerido

- **Lenguaje:** Python 3.11+
- **Orquestación RAG:** LangChain o LlamaIndex (elegir uno; LlamaIndex es más directo para RAG puro)
- **Embeddings:** `text-embedding-3-small` (OpenAI) o `bge-small`/`e5` (open source vía `sentence-transformers`) para una variante sin costo
- **Vector store:** Chroma (local, cero fricción para empezar) con opción de migrar a Qdrant (producción, filtros, escalado)
- **LLM de generación:** OpenAI / Anthropic / un modelo local vía Ollama
- **Evaluación:** RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) y/o DeepEval
- **API:** FastAPI (endpoint `/query`)
- **Ingesta/parseo:** `unstructured`, `pypdf`, `markdown` según el formato del corpus
- **Testing / calidad:** pytest, ruff, black

## Estructura de carpetas propuesta

Sigue el **src layout con paquete nombrado** (estándar de mercado para un paquete instalable):

```
rag-pipeline-eval/
├── README.md
├── AGENTS.md                     # Ruleset ponytail (código mínimo)
├── pyproject.toml
├── .env.example
├── data/
│   ├── raw/                      # Corpus original (PDF, MD, HTML, TXT)
│   ├── interim/                  # Intermedios de procesamiento
│   ├── processed/                # Vector store (Chroma) — git-ignored
│   └── external/                 # Datos de terceros
├── src/
│   └── rag_pipeline_eval/        # Paquete importable
│       ├── config.py            # Settings (pydantic-settings): modelos, chunk size, top_k
│       ├── ingestion/
│       │   ├── loaders.py       # Carga por tipo de documento
│       │   └── chunking.py      # Estrategias de chunking (sentence, semantic)
│       ├── embeddings/
│       │   └── embedder.py      # Modelo de embeddings (factory intercambiable)
│       ├── vectorstore/
│       │   └── store.py         # Abstracción Chroma/Qdrant (interfaz común)
│       ├── retrieval/
│       │   └── retriever.py     # Índice + búsqueda por similitud
│       ├── generation/
│       │   └── rag_chain.py     # Prompt + recuperación + LLM (Claude)
│       └── api/
│           └── main.py          # FastAPI: POST /query
├── scripts/
│   └── index.py                 # Indexación idempotente (CLI)
├── eval/
│   ├── datasets/
│   │   └── qa_golden.jsonl      # Preguntas + respuestas de referencia
│   ├── run_eval.py              # Corre RAGAS y genera el reporte
│   └── reports/                 # Salidas de evaluación (JSON/CSV) versionadas
├── notebooks/                   # Exploración y tuning de chunking/top_k
└── tests/                       # Espejan src/: tests/ingestion/, etc.
    ├── ingestion/
    │   └── test_chunking.py
    └── test_config.py
```

## Criterios de "terminado"

Se puede indexar un corpus, consultar vía API y **regenerar un reporte de evaluación con métricas cuantitativas** que respalden las decisiones de diseño. La calidad del RAG está medida, no asumida.

---

## Puesta en marcha (scaffold generado)

Stack de arranque: **LlamaIndex** + **embeddings open source** (sentence-transformers, corren en local) + **Claude** para la generación + **Chroma** como vector store.

### 1. Instalar dependencias (Poetry)

```bash
poetry install                 # dependencias base
poetry install --with eval,dev # + evaluación (RAGAS) y herramientas de dev
```

### 2. Configurar el entorno

```bash
cp .env.example .env
# Editar .env y completar ANTHROPIC_API_KEY
```

### 3. Indexar el corpus

Dejá tus documentos en `data/raw/` (ya hay dos de ejemplo) y corré:

```bash
poetry run python -m scripts.index          # idempotente
poetry run python -m scripts.index --force  # re-indexar
```

### 4. Levantar la API y consultar

```bash
poetry run uvicorn rag_pipeline_eval.api.main:app --reload
# En otra terminal:
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "¿Qué mide la métrica faithfulness?"}'
```

### 5. Evaluar la calidad

```bash
poetry run python -m eval.run_eval --tag baseline
# Reporte en eval/reports/baseline-<timestamp>.json
```

### Tests

```bash
poetry run pytest
```

> El código está organizado por responsabilidad en `src/rag_pipeline_eval/`
> (ingestion, embeddings, vectorstore, retrieval, generation, api), con interfaces
> intercambiables para poder comparar configuraciones — que es el insumo del
> módulo de evaluación.
