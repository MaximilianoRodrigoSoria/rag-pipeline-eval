"""Configuración central del pipeline RAG.

Todas las settings se leen desde variables de entorno (o un archivo `.env`)
mediante pydantic-settings, de modo que no haya valores mágicos dispersos por
el código. Ver `.env.example` para la lista completa.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del proyecto (…/rag-pipeline-eval)
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Settings tipadas del pipeline. Se instancian una sola vez (ver `get_settings`)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Generación (Claude) ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="claude-3-5-sonnet-latest", alias="LLM_MODEL")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    # --- Embeddings (open source) ---
    embed_model: str = Field(default="intfloat/multilingual-e5-small", alias="EMBED_MODEL")

    # --- Chunking ---
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    chunk_strategy: Literal["sentence", "semantic"] = Field(
        default="sentence", alias="CHUNK_STRATEGY"
    )

    # --- Retrieval ---
    top_k: int = Field(default=4, alias="TOP_K")

    # --- Vector store (Chroma) ---
    chroma_path: str = Field(default="./data/processed/chroma", alias="CHROMA_PATH")
    chroma_collection: str = Field(default="rag_corpus", alias="CHROMA_COLLECTION")

    # --- Datos ---
    corpus_dir: str = Field(default="./data/raw", alias="CORPUS_DIR")

    # --- Paths resueltos (absolutos, relativos a la raíz del proyecto) ---
    def path(self, value: str) -> Path:
        """Convierte una ruta relativa de config en absoluta respecto a la raíz."""
        p = Path(value)
        return p if p.is_absolute() else (ROOT_DIR / p)

    @property
    def chroma_dir(self) -> Path:
        return self.path(self.chroma_path)

    @property
    def corpus_path(self) -> Path:
        return self.path(self.corpus_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la instancia única de settings (cacheada)."""
    return Settings()
