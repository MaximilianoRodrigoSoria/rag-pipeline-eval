"""Estrategias de chunking (troceado de documentos en nodos).

Exponemos dos estrategias intercambiables detrás de una misma interfaz:

- ``sentence``: `SentenceSplitter`, rápido y determinístico. Divide respetando
  límites de oración con un tamaño/solape configurables. Buen default.
- ``semantic``: `SemanticSplitterNodeParser`, agrupa oraciones por similitud de
  embeddings (fronteras "semánticas"). Más costoso porque embebe al trocear,
  pero suele mejorar la precisión del contexto.

Comparar ambas estrategias en el módulo de evaluación es justamente uno de los
diferenciadores del proyecto.
"""

from __future__ import annotations

from llama_index.core.node_parser import (
    NodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import BaseNode, Document

from src.config import Settings


def build_node_parser(settings: Settings, embed_model=None) -> NodeParser:
    """Construye el node parser según la estrategia configurada.

    Args:
        settings: configuración (estrategia, chunk_size, chunk_overlap).
        embed_model: modelo de embeddings, requerido solo para la estrategia
            ``semantic``.

    Returns:
        Un `NodeParser` de LlamaIndex listo para trocear documentos.
    """
    if settings.chunk_strategy == "semantic":
        if embed_model is None:
            raise ValueError("La estrategia 'semantic' requiere un embed_model.")
        return SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )

    # Estrategia por defecto: sentence
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def chunk_documents(
    documents: list[Document], settings: Settings, embed_model=None
) -> list[BaseNode]:
    """Trocea una lista de documentos en nodos según la estrategia elegida."""
    parser = build_node_parser(settings, embed_model=embed_model)
    return parser.get_nodes_from_documents(documents)
