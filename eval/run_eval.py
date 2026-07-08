"""Evaluación automatizada del RAG con RAGAS.

Corre el pipeline sobre el dataset dorado (`qa_golden.jsonl`), recolecta para
cada pregunta la respuesta generada y los contextos recuperados, y calcula las
métricas de RAGAS: faithfulness, answer_relevancy, context_precision,
context_recall.

El LLM juez sigue al proveedor configurado (`LLM_PROVIDER`): Claude (Anthropic)
u Ollama local. Genera un reporte reproducible en `eval/reports/` (JSON + CSV).

Uso:
    python -m eval.run_eval
    python -m eval.run_eval --dataset eval/datasets/qa_golden.jsonl --tag baseline
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rag_pipeline_eval.config import ROOT_DIR, get_settings
from rag_pipeline_eval.generation.rag_chain import RagEngine


def load_golden(path: Path) -> list[dict]:
    """Carga el dataset dorado en formato JSONL."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Dataset vacío: {path}")
    return rows


def collect_samples(engine: RagEngine, golden: list[dict]) -> dict[str, list]:
    """Corre el RAG sobre cada pregunta y arma las columnas que espera RAGAS."""
    questions, answers, contexts, ground_truths = [], [], [], []
    for row in golden:
        result = engine.query(row["question"])
        questions.append(row["question"])
        answers.append(result.answer)
        contexts.append([s["text"] for s in result.sources])
        ground_truths.append(row.get("ground_truth", ""))
    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def _judge_llm(settings):
    """Devuelve el LLM juez según el proveedor configurado (imports diferidos)."""
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
        )
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=0.0,
        max_tokens=settings.llm_max_tokens,
    )


def run_ragas(samples: dict[str, list], settings) -> "object":
    """Evalúa con RAGAS usando el juez configurado y embeddings open source."""
    from datasets import Dataset
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict(samples)
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=_judge_llm(settings),
        embeddings=HuggingFaceEmbeddings(model_name=settings.embed_model),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa el RAG con RAGAS.")
    parser.add_argument("--dataset", default="eval/datasets/qa_golden.jsonl")
    parser.add_argument("--tag", default="baseline", help="Etiqueta de la corrida.")
    args = parser.parse_args()

    settings = get_settings()
    golden = load_golden(ROOT_DIR / args.dataset)
    print(f"Evaluando {len(golden)} preguntas (tag: {args.tag}) …")

    engine = RagEngine(settings)
    samples = collect_samples(engine, golden)

    print(f"Calculando métricas con RAGAS (juez: {settings.llm_provider}) …")
    result = run_ragas(samples, settings)
    scores = result.to_pandas()

    reports_dir = ROOT_DIR / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{args.tag}-{ts}"

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = {
        "tag": args.tag,
        "timestamp": ts,
        "config": {
            "llm_provider": settings.llm_provider,
            "embed_model": settings.embed_model,
            "chunk_strategy": settings.chunk_strategy,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
        },
        "metrics": {c: float(scores[c].mean()) for c in metric_cols if c in scores},
        "n_questions": len(golden),
    }

    (reports_dir / f"{stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    scores.to_csv(reports_dir / f"{stem}.csv", index=False)

    print("\n=== Resultados (promedios) ===")
    for metric, value in summary["metrics"].items():
        print(f"  {metric:20s}: {value:.3f}")
    print(f"\nReporte guardado en eval/reports/{stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
