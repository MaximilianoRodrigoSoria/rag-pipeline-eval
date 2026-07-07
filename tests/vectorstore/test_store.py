"""Round-trip real contra Chroma temporal, con embeddings mockeados."""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("llama_index.vector_stores.chroma")

from rag_pipeline_eval.ingestion.chunking import chunk_documents  # noqa: E402
from rag_pipeline_eval.retrieval import retriever as retr  # noqa: E402
from rag_pipeline_eval.vectorstore.store import collection_count  # noqa: E402


def test_index_and_retrieve(settings, sample_docs, mock_embed, monkeypatch):
    # Evita cargar el modelo HF real: el retriever usa el embed mockeado.
    monkeypatch.setattr(retr, "get_embed_model", lambda s: mock_embed)

    nodes = chunk_documents(sample_docs, settings, embed_model=mock_embed)
    retr.build_index(nodes, settings)
    assert collection_count(settings) > 0

    results = retr.get_retriever(settings).retrieve("¿qué mide faithfulness?")
    assert len(results) > 0
