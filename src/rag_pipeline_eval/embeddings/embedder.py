"""Modelo de embeddings.

Se aísla la creación del embedder detrás de una función factory para que sea
intercambiable: hoy usamos un modelo open source (sentence-transformers vía
HuggingFace) que corre en local sin costo ni API key; mañana podría cambiarse
por OpenAI u otro sin tocar el resto del pipeline.
"""

from __future__ import annotations

from functools import lru_cache

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag_pipeline_eval.config import Settings


@lru_cache(maxsize=2)
def _cached_hf_embedding(model_name: str) -> BaseEmbedding:
    """Cachea el modelo HF por nombre (cargarlo es caro)."""
    return HuggingFaceEmbedding(model_name=model_name)


def get_embed_model(settings: Settings) -> BaseEmbedding:
    """Devuelve el modelo de embeddings configurado.

    Args:
        settings: configuración con `embed_model`.

    Returns:
        Un embedding compatible con LlamaIndex.
    """
    return _cached_hf_embedding(settings.embed_model)
