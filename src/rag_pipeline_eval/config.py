"""Configuración central del pipeline RAG.

Todas las settings se leen desde variables de entorno (o un archivo `.env`)
mediante pydantic-settings. Ver `.env.example` para la lista completa.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del proyecto (…/rag-pipeline-eval)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Settings tipadas del pipeline. Se instancian una sola vez (ver `get_settings`)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Permite construir Settings(campo=...) por nombre además de por alias/env var
        # (necesario para overridear settings en tests).
        populate_by_name=True,
    )

    # --- Proveedor de generación ---
    # "anthropic" (Claude, por API) o "ollama" (LLM local). Conmutable por .env.
    llm_provider: Literal["anthropic", "ollama"] = Field(
        default="anthropic", alias="LLM_PROVIDER"
    )

    # --- Generación (Claude) ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="claude-3-5-sonnet-latest", alias="LLM_MODEL")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    # --- Generación (Ollama, local) ---
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # --- Juez de evaluación (RAGAS) ---
    # Desacopla el juez de la generación: se puede generar local (Ollama) y juzgar
    # con Claude para métricas más confiables. Vacío = usa el mismo `llm_provider`.
    judge_provider: Literal["", "anthropic", "ollama"] = Field(
        default="", alias="JUDGE_PROVIDER"
    )

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

    @property
    def effective_judge_provider(self) -> str:
        """Proveedor del juez de eval: `judge_provider` si se setea, si no `llm_provider`."""
        return self.judge_provider or self.llm_p