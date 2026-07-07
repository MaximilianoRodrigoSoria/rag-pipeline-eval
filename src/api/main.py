"""API FastAPI del pipeline RAG.

Expone `POST /query` que recibe una pregunta y devuelve la respuesta generada
más las fuentes recuperadas. El motor RAG se inicializa de forma perezosa en el
primer request (para no cargar el índice/LLM al importar el módulo).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings
from src.generation.rag_chain import RagEngine
from src.vectorstore.store import collection_count

app = FastAPI(
    title="rag-pipeline-eval",
    description="Pipeline RAG sobre corpus propio con evaluación automatizada.",
    version="0.1.0",
)

_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    """Inicializa el motor RAG una sola vez (lazy)."""
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


@app.get("/health")
def health() -> dict:
    """Chequeo de salud + cantidad de vectores indexados."""
    settings = get_settings()
    return {"status": "ok", "indexed_vectors": collection_count(settings)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Responde una pregunta usando el corpus indexado."""
    try:
        result = get_engine().query(req.question)
    except RuntimeError as exc:  # p. ej. falta API key
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QueryResponse(answer=result.answer, sources=result.sources)
