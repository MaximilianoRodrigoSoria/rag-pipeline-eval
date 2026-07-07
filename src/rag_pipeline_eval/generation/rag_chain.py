"""Cadena RAG: recuperación + generación con Claude.

Arma un query engine sobre el índice con un prompt de sistema anti-alucinación
que obliga al modelo a responder SOLO con el contexto recuperado y a citar las
fuentes. Devuelve la respuesta junto con los fragmentos usados, de modo que la
salida sea auditable y sirva de insumo para la evaluación (faithfulness, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.llms.anthropic import Anthropic

from rag_pipeline_eval.config import Settings
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


def _build_llm(settings: Settings) -> Anthropic:
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "Falta ANTHROPIC_API_KEY. Copiá .env.example a .env y completá la key."
        )
    return Anthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )


class RagEngine:
    """Motor RAG reutilizable (carga el índice y el LLM una sola vez)."""

    def __init__(self, settings: Settings, index: VectorStoreIndex | None = None):
        self.settings = settings
        self.index = index or load_index(settings)
        self.llm = _build_llm(settings)
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
