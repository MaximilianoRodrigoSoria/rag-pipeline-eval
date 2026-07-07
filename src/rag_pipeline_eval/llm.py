"""Factory del LLM de generación.

Aísla la construcción del LLM detrás de una función para poder cambiar de
proveedor por configuración (Claude por API hoy; Ollama local cuando se pueda)
sin tocar el resto del pipeline. Es, además, el punto por donde los tests
inyectan un LLM mockeado.
"""

from __future__ import annotations

from rag_pipeline_eval.config import Settings


def get_llm(settings: Settings):
    """Devuelve el LLM según `settings.llm_provider`. Imports diferidos para no
    exigir el SDK del proveedor que no se usa."""
    if settings.llm_provider == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            request_timeout=120.0,
            temperature=settings.llm_temperature,
        )

    # Default: Anthropic (Claude)
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "Falta ANTHROPIC_API_KEY. Completá el .env, o poné LLM_PROVIDER=ollama "
            "para correr con un modelo local."
        )
    from llama_index.llms.anthropic import Anthropic

    return Anthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
