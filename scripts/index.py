"""Script de indexación idempotente.

Carga el corpus, lo trocea según la estrategia configurada, y persiste los
embeddings en Chroma. Es idempotente: si la colección ya tiene vectores, no
re-indexa salvo que se pase ``--force`` (que borra la colección y la reconstruye
desde cero, evitando vectores duplicados).

Uso:
    python -m scripts.index
    python -m scripts.index --force
"""

from __future__ import annotations

import argparse
import sys

from rag_pipeline_eval.config import get_settings
from rag_pipeline_eval.embeddings.embedder import get_embed_model
from rag_pipeline_eval.ingestion.chunking import chunk_documents
from rag_pipeline_eval.ingestion.loaders import load_documents
from rag_pipeline_eval.retrieval.retriever import build_index
from rag_pipeline_eval.vectorstore.store import collection_count, reset_collection


def main() -> int:
    parser = argparse.ArgumentParser(description="Indexa el corpus en el vector store.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-indexa aunque ya existan vectores.",
    )
    args = parser.parse_args()

    settings = get_settings()

    existing = collection_count(settings)
    if existing > 0 and not args.force:
        print(
            f"La colección '{settings.chroma_collection}' ya tiene {existing} vectores. "
            "Usá --force para re-indexar."
        )
        return 0

    if args.force and existing > 0:
        print(f"--force: borrando {existing} vectores previos …")
        reset_collection(settings)

    print(f"Cargando corpus desde {settings.corpus_path} …")
    documents = load_documents(settings.corpus_path)
    print(f"  {len(documents)} documento(s) cargado(s).")

    embed_model = get_embed_model(settings)
    print(f"Troceando (estrategia: {settings.chunk_strategy}) …")
    nodes = chunk_documents(documents, settings, embed_model=embed_model)
    print(f"  {len(nodes)} nodo(s) generado(s).")

    print("Indexando en Chroma (esto genera los embeddings) …")
    build_index(nodes, settings)
    print(f"Listo. Vectores en la colección: {collection_count(settings)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
