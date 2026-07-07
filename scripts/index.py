"""Script de indexación idempotente.

Carga el corpus, lo trocea según la estrategia configurada, y persiste los
embeddings en Chroma. Es idempotente: si la colección ya tiene vectores, no
re-indexa salvo que se pase ``--force`` (que reconstruye desde cero usando una
colección temporal, para no romper el estado previo en mounts que no permiten
borrado).

Uso:
    python -m scripts.index
    python -m scripts.index --force
"""

from __future__ import annotations

import argparse
import sys

from src.config import get_settings
from src.embeddings.embedder import get_embed_model
from src.ingestion.chunking import chunk_documents
from src.ingestion.loaders import load_documents
from src.retrieval.retriever import build_index
from src.vectorstore.store import collection_count


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
