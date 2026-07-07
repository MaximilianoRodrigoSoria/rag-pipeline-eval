"""Abstracción del vector store (implementación Chroma).

Se expone una interfaz mínima y estable (`get_vector_store`, `get_storage_context`)
para que el resto del pipeline no dependa de Chroma directamente. Migrar a Qdrant
en el futuro implicaría solo agregar otra implementación detrás de esta interfaz.

Chroma se usa en modo persistente (PersistentClient), de manera que el índice
sobrevive entre corridas y la indexación puede ser idempotente.
"""

from __future__ import annotations

import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_pipeline_eval.config import Settings


def get_chroma_collection(settings: Settings):
    """Devuelve (o crea) la colección de Chroma persistente."""
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(name=settings.chroma_collection)


def get_vector_store(settings: Settings) -> ChromaVectorStore:
    """Devuelve el `ChromaVectorStore` de LlamaIndex sobre la colección persistente."""
    collection = get_chroma_collection(settings)
    return ChromaVectorStore(chroma_collection=collection)


def get_storage_context(settings: Settings) -> StorageContext:
    """StorageContext que envuelve el vector store (para construir/cargar el índice)."""
    return StorageContext.from_defaults(vector_store=get_vector_store(settings))


def collection_count(settings: Settings) -> int:
    """Cantidad de vectores ya indexados (útil para chequear estado)."""
    return get_chroma_collection(settings).count()


def reset_collection(settings: Settings) -> None:
    """Borra la colección para re-indexar desde cero (usado por --force).

    Evita el footgun de re-indexar sobre una colección poblada (duplicaría
    vectores). Si no existe, no hay nada que borrar.
    """
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(name=settings.chroma_collection)
    except Exception:  # ponytail: delete_collection tira si no existe; es idempotente a propósito
        pass
