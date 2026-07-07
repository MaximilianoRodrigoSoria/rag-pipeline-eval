"""El motor RAG responde y adjunta fuentes, todo con mocks (sin API ni red)."""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core")

from rag_pipeline_eval.generation.rag_chain import RagAnswer, RagEngine  # noqa: E402


def test_engine_query_with_mocks(settings, in_memory_index, mock_llm):
    engine = RagEngine(settings, index=in_memory_index, llm=mock_llm)
    ans = engine.query("¿qué es Chroma?")

    assert isinstance(ans, RagAnswer)
    assert isinstance(ans.answer, str) and ans.answer
    assert len(ans.sources) > 0
    assert "source" in ans.sources[0]
