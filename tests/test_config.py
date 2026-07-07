"""Tests de configuración (no requieren dependencias pesadas)."""

from __future__ import annotations

from pathlib import Path

from src.config import ROOT_DIR, Settings


def test_defaults_razonables():
    s = Settings(_env_file=None)
    assert s.chunk_size > s.chunk_overlap
    assert s.top_k >= 1
    assert s.chunk_strategy in ("sentence", "semantic")


def test_path_resuelve_absoluto():
    s = Settings(_env_file=None)
    assert s.chroma_dir.is_absolute()
    assert s.corpus_path.is_absolute()
    # Las rutas relativas cuelgan de la raíz del proyecto
    assert str(s.corpus_path).startswith(str(ROOT_DIR))


def test_path_absoluto_se_respeta():
    s = Settings(_env_file=None)
    abs_path = Path("/tmp/x").resolve()
    assert s.path(str(abs_path)) == abs_path
