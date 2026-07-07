"""Cadena RAG: recuperación + generación con el LLM configurado.

El LLM se inyecta (o se resuelve por config vía `get_llm`), de modo que:
- se puede cambiar de proveedor (Claude API / Ollama local) por `.env`, y
- los tests pueden pasar un LLM mockeado sin API key ni modelo real.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llama_index.core import PromptTemplate, VectorStoreIndex

from rag_pipeline_eval.config import Settings
from rag_pipeline_eval.llm import get_llm
from rag_pipeline_eval.retrieval.retriever import load_index

# Prompt con guardas explícitas: sin invención, y si no hay soporte, decirlo.
QA_PROMPT = PromptTemplate(
    "Sos un asistente que responde ÚNICAMENTE con la información del contexto.\n"
    "Reglas:\n"
    "1. No inventes datos: si el contexto no alcanza para responder, decí "
    "explícitamente que no hay información suficiente.\n"
    "2. Citá las fuentes usadas al final, entre corchetes, con el campo 'source'.\n"
    "3. Sé conciso y preciso.\n\n"
    "----------------\n"
    "Contexto:\n{context_str}\n"
    "----------------\n"
    "Pregunta: {query_str}\n"
    "Respuesta:"
)


@dataclass
class RagAnswer:
    """Resultado de una consulta RAG."""

    answer: str
    sources: list[dict] = field(default_factory=list)


class RagEngine:
    """Motor RAG reutilizable (carga índice y LLM una sola vez).

    Args:
        settings: configuración.
        index: índice ya construido (opcional; si falta se carga de Chroma).
        llm: LLM a usar (opcional; si falta se resuelve por config). Este es el
            seam para inyectar Claude/Ollama real o un mock en tests.
    """

    def __init__(self, settings: Settings, index: VectorStoreIndex | None = None, llm=None):
        self.settings = settings
        self.index = index or load_index(settings)
        self.llm = llm or get_llm(settings)
        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=settings.top_k,
            text_qa_template=QA_PROMPT,
        )

    def query(self, question: str) -> RagAnswer:
        """Responde una pregunta y adjunta las fuentes recuperadas."""
        response = self.query_engine.query(question)
        sources = []
        for node in response.source_nodes:
            sources.append(
                {
                    "source": node.metadata.get("source", node.node_id),
                    "score": round(float(node.score), 4) if node.score is not None else None,
                    "text": node.get_content()[:500],
                }
            )
        return RagAnswer(answer=str(response), sources=sources)
