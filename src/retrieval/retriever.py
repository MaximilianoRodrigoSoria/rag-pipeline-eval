"""Construcción del índice y del retriever.

Reúne las piezas (embeddings + vector store) para:

- ``build_index``: indexar los nodos troceados en Chroma (usado por el script de
  indexación).
- ``load_index``: reconstruir el `VectorStoreIndex` desde Chroma ya poblado (usado
  en query time, sin re-indexar).
- ``get_retriever``: devolver un retriever configurado con `top_k`.
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode

from src.config import Settings
from src.embeddings.embedder import get_embed_model
from src.vectorstore.store import get_storage_context, get_vector_store


def build_index(nodes: list[BaseNode], settings: Settings) -> VectorStoreIndex:
    """Indexa los nodos en el vector store y devuelve el índice."""
    embed_model = get_embed_model(settings)
    storage_context = get_storage_context(settings)
    return VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )


def load_index(settings: Settings) -> VectorStoreIndex:
    """Carga el índice desde un Chroma ya poblado (sin re-indexar)."""
    embed_model = get_embed_model(settings)
    vector_store = get_vector_store(settings)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )


def get_retriever(settings: Settings, index: VectorStoreIndex | None = None) -> BaseRetriever:
    """Devuelve un retriever por similitud con `top_k` del config."""
    if index is None:
        index = load_index(settings)
    return index.as_retriever(similarity_top_k=settings.top_k)
