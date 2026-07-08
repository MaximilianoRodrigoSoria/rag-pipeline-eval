"""Smoke test del juez de evaluación.

Hace UNA sola llamada al LLM juez configurado (según `JUDGE_PROVIDER` /
`LLM_PROVIDER`) para verificar que el modelo y la API key resuelven bien,
antes de lanzar la evaluación completa (que hace decenas de llamadas).

Uso:
    set PYTHONPATH=src
    python -m scripts.check_judge
"""

from __future__ import annotations

import sys

from rag_pipeline_eval.config import get_settings

# Reutiliza la misma lógica de selección de juez que la evaluación.
sys.path.insert(0, "eval")
from run_eval import _judge_llm  # noqa: E402


def main() -> int:
    settings = get_settings()
    provider = settings.effective_judge_provider
    model = settings.effective_judge_model if provider == "anthropic" else settings.ollama_model
    print(f"Juez: {provider} · modelo: {model}")

    try:
        llm = _judge_llm(settings)
        resp = llm.invoke("Responde únicamente con la palabra: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ El juez NO responde: {type(exc).__name__}: {exc}")
        print("   Revisá LLM_MODEL / ANTHROPIC_API_KEY (o que Ollama esté levantado).")
        return 1

    content = getattr(resp, "content", resp)
    print(f"✅ El juez respondió: {content!r}")
    print("   Podés correr la evaluación completa con:  python -m eval.run_eval --tag baseline-claude-judge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
