"""Tests de la API con TestClient, overrideando el motor vía dependency_overrides."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from rag_pipeline_eval.api.main import app, get_engine  # noqa: E402
from rag_pipeline_eval.generation.rag_chain import RagAnswer  # noqa: E402


class _StubEngine:
    def query(self, question: str) -> RagAnswer:
        return RagAnswer(
            answer=f"stub: {question}",
            sources=[{"source": "x.md", "score": 0.9, "text": "contexto"}],
        )


def test_query_endpoint_ok():
    app.dependency_overrides[get_engine] = lambda: _StubEngine()
    try:
        client = TestClient(app)
        resp = client.post("/query", json={"question": "hola"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"].startswith("stub:")
        assert body["sources"][0]["source"] == "x.md"
    finally:
        app.dependency_overrides.clear()


def test_query_validation_422():
    client = TestClient(app)
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422
