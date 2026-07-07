"""API FastAPI del pipeline RAG.

Buenas prácticas aplicadas (equivalentes a features de Spring):
- Inyección de dependencias con `Depends(get_engine)` (patrón DI), lo que permite
  overridear el motor en tests (`app.dependency_overrides`).
- Endpoint async + threadpool para la llamada bloqueante al LLM (patrón `@Async`).
- Exception handler global que traduce errores de config a HTTP (patrón
  `@ControllerAdvice`).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_pipeline_eval.config import get_settings
from rag_pipeline_eval.generation.rag_chain import RagEngine
from rag_pipeline_eval.vectorstore.store import collection_count

app = FastAPI(
    title="rag-pipeline-eval",
    description="Pipeline RAG sobre corpus propio con evaluación automatizada.",
    version="0.1.0",
)

_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    """Provee el motor RAG (lazy singleton). Se puede overridear en tests."""
    global _engine
    if _engine is None:
        _engine = RagEngine(get_settings())
    return _engine


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pregunta del usuario.")


class Source(BaseModel):
    source: str
    score: float | None = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """Traduce fallos de configuración/entorno (p. ej. falta API key) a 503."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict:
    """Chequeo de salud + cantidad de vectores indexados."""
    return {"status": "ok", "indexed_vectors": collection_count(get_settings())}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, engine: RagEngine = Depends(get_engine)) -> QueryResponse:
    """Responde una pregunta usando el corpus indexado.

    La query de LlamaIndex es bloqueante (I/O + LLM); la corremos en un threadpool
    para no bloquear el event loop.
    """
    result = await run_in_threadpool(lambda: engine.query(req.question))
    return QueryResponse(answer=result.answer, sources=result.sources)
