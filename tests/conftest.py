"""Fixtures compartidas de la suite.

Todo se ejecuta OFFLINE: se usan MockLLM y MockEmbedding de LlamaIndex, así los
tests corren sin API key ni descargar el modelo de embeddings. El seam para
"meter la API/LLM real cuando se pueda" es inyectar el LLM real en RagEngine o
poner LLM_PROVIDER en el .env.
"""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core import Document, VectorStoreIndex  # noqa: E402
from llama_index.core.embeddings.mock_embed_model import MockEmbedding  # noqa: E402
from llama_index.core.llms import MockLLM  # noqa: E402

from rag_pipeline_eval.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    """Settings apuntando a un Chroma temporal (aislado por test)."""
    return Settings(
        _env_file=None,
        chroma_path=str(tmp_path / "chroma"),
        corpus_dir=str(tmp_path / "raw"),
        top_k=2,
    )


@pytest.fixture
def mock_embed():
    """Embeddings falsos, deterministas, sin descargar nada."""
    return MockEmbedding(embed_dim=8)


@pytest.fixture
def mock_llm():
    """LLM falso: responde algo fijo, suficiente para ejercitar el pipeline."""
    return MockLLM(max_tokens=64)


@pytest.fixture
def sample_docs():
    return [
        Document(
            text="El faithfulness mide si la respuesta se sostiene en el contexto recuperado.",
            metadata={"source": "a.md"},
        ),
        Document(
            text="Chroma es un vector store local para búsqueda por similitud.",
            metadata={"source": "b.md"},
        ),
    ]


@pytest.fixture
def in_memory_index(sample_docs, mock_embed):
    """Índice en memoria (sin Chroma) para tests de generación."""
    return VectorStoreIndex.from_documents(sample_docs, embed_model=mock_embed)
