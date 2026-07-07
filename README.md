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
- [Checklist de implementación](#checklist-de-implementación)
- [Puesta en marcha (scaffold generado)](#puesta-en-marcha-scaffold-generado)

</details>

## Objetivo

Construir un RAG end-to-end, reproducible y medible: ingerir documentos propios, trocearlos (chunking), generar embeddings, almacenarlos en un vector store, recuperar contexto relevante y generar respuestas fundamentadas. El diferencial del proyecto no es "que responda", sino **demostrar con métricas** que responde bien: sin alucinar (faithfulness), atendiendo a la pregunta (answer relevancy) y recuperando el contexto correcto (context precision/recall).

El resultado es un servicio consultable (API o CLI) más un reporte de evaluación que se puede regenerar en cada cambio del pipeline, sirviendo de base para los demás proyectos del portfolio.

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

## Checklist de implementación

### Fase 1 — Setup

- [ ] Inicializar el proyecto (`uv`/`poetry`), `pyproject.toml`, `.env.example`, `pre-commit`.
- [ ] Definir `config.py` con settings tipadas (modelo de embeddings, LLM, `chunk_size`, `chunk_overlap`, `top_k`).
- [ ] Elegir y documentar el corpus propio (ej. documentación técnica, papers, notas). Dejarlo en `data/raw/`.

### Fase 2 — Ingesta y chunking

- [ ] Implementar loaders por tipo de documento en `ingestion/loaders.py`.
- [ ] Implementar al menos dos estrategias de chunking (recursive y semantic) en `chunking.py`.
- [ ] Persistir chunks + metadatos (source, page, chunk_id) en `data/processed/`.

### Fase 3 — Embeddings y vector store

- [ ] Imple