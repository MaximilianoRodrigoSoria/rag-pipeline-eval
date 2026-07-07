"""Tests del troceado (chunking).

No requieren API key ni el LLM: validan que la estrategia por defecto (sentence)
produzca nodos y respete aproximadamente el tamaño configurado.
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.ingestion.chunking import build_node_parser, chunk_documents

pytest.importorskip("llama_index.core", reason="requiere llama-index instalado")

from llama_index.core.schema import Document  # noqa: E402


def _settings(**overrides) -> Settings:
    base = dict(chunk_size=128, chunk_overlap=16, chunk_strategy="sentence")
    base.update(overrides)
    return Settings(**base)


def test_sentence_parser_genera_nodos():
    settings = _settings()
    text = "Una oración. " * 200
    nodes = chunk_documents([Document(text=text)], settings)
    assert len(nodes) > 1, "un texto largo debería producir varios nodos"


def test_nodos_conservan_metadata_source():
    settings = _settings()
    doc = Document(text="Contenido de prueba. " * 50, metadata={"source": "doc.md"})
    nodes = chunk_documents([doc], settings)
    assert all(n.metadata.get("source") == "doc.md" for n in nodes)


def test_semantic_requiere_embed_model():
    settings = _settings(chunk_strategy="semantic")
    with pytest.raises(ValueError):
        build_node_parser(settings, embed_model=None)
