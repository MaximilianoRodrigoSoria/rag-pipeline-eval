# rag-pipeline-eval

Pipeline de **Retrieval-Augmented Generation** sobre un corpus propio, con un módulo de **evaluación automatizada** que mide la calidad de las respuestas (faithfulness y relevancia) en lugar de asumirla.

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

```
rag-pipeline-eval/
├── README.md
├── pyproject.toml
├── .env.example
├── data/
│   ├── raw/                  # Corpus original (PDF, MD, HTML, TXT)
│   └── processed/            # Chunks y metadatos intermedios
├── src/
│   ├── config.py            # Settings (pydantic-settings): modelos, chunk size, top_k
│   ├── ingestion/
│   │   ├── loaders.py       # Carga por tipo de documento
│   │   └── chunking.py      # Estrategias de chunking (fixed, recursive, semantic)
│   ├── embeddings/
│   │   └── embedder.py      # Wrapper del modelo de embeddings
│   ├── vectorstore/
│   │   └── store.py         # Abstracción Chroma/Qdrant (interfaz común)
│   ├── retrieval/
│   │   └── retriever.py     # Búsqueda por similitud + reranking opcional
│   ├── generation/
│   │   └── rag_chain.py     # Prompt + recuperación + LLM
│   └── api/
│       └── main.py          # FastAPI: POST /query
├── eval/
│   ├── datasets/
│   │   └── qa_golden.jsonl  # Preguntas + respuestas/contexto de referencia
│   ├── run_eval.py          # Corre RAGAS/DeepEval y genera el reporte
│   └── reports/             # Salidas de evaluación (JSON/HTML/MD) versionadas
├── notebooks/
│   └── explore.ipynb        # Exploración y tuning de chunking/top_k
└── tests/
    ├── test_chunking.py
    └── test_retrieval.py
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

- [ ] Implementar el `embedder` con interfaz intercambiable (OpenAI vs open source).
- [ ] Implementar la abstracción de vector store con una interfaz común y una implementación Chroma.
- [ ] Script de indexación idempotente (re-indexar sin duplicar).
- [ ] (Opcional) Añadir implementación Qdrant y comparar.

### Fase 4 — Retrieval y generación

- [ ] Implementar el retriever (similarity search + filtros por metadato).
- [ ] (Opcional) Añadir un reranker (`cross-encoder` / Cohere Rerank) y medir su impacto.
- [ ] Diseñar el prompt del `rag_chain` con instrucciones anti-alucinación y cita de fuentes.
- [ ] Exponer `POST /query` en FastAPI devolviendo respuesta + fuentes recuperadas.

### Fase 5 — Evaluación (el diferenciador)

- [ ] Construir el dataset dorado `qa_golden.jsonl` (mínimo 20–30 preguntas representativas).
- [ ] Integrar RAGAS: faithfulness, answer_relevancy, context_precision, context_recall.
- [ ] `run_eval.py` que genere un reporte reproducible en `eval/reports/`.
- [ ] Comparar al menos dos configuraciones (ej. chunk size A vs B, con/sin reranker) y documentar cuál gana y por qué.
- [ ] (Opcional) Fijar umbrales mínimos y hacer que la evaluación falle si se cruzan (base para CI).

### Fase 6 — Documentación

- [ ] README con instrucciones de setup, indexación y consulta.
- [ ] ADR breve: elección de vector store y estrategia de chunking.
- [ ] Tabla de resultados de evaluación en el README con la configuración final.

## Criterios de "terminado"

Se puede indexar un corpus, consultar vía API y **regenerar un reporte de evaluación con métricas cuantitativas** que respalden las decisiones de diseño. La calidad del RAG está medida, no asumida.
