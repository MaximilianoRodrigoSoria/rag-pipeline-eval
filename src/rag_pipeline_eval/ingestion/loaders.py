"""Carga de documentos del corpus.

Usa `SimpleDirectoryReader` de LlamaIndex, que ya resuelve la lectura por tipo
de archivo (PDF, MD, TXT, HTML, DOCX, …) y adjunta metadatos básicos como el
nombre del archivo de origen. Sobre eso agregamos un `source` estable que luego
sirve para citar y para las métricas de evaluación.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

# Extensiones soportadas por defecto (SimpleDirectoryReader trae más lectores
# si se instalan las dependencias correspondientes).
SUPPORTED_EXTS = [".md", ".txt", ".pdf", ".html", ".htm", ".docx"]


def load_documents(corpus_dir: str | Path) -> list[Document]:
    """Lee todos los documentos del directorio de corpus.

    Args:
        corpus_dir: carpeta con los documentos crudos (data/raw por defecto).

    Returns:
        Lista de `Document` de LlamaIndex, cada uno con `metadata["source"]`.

    Raises:
        FileNotFoundError: si el directorio no existe.
        ValueError: si no hay ningún documento soportado.
    """
    corpus = Path(corpus_dir)
    if not corpus.exists():
        raise FileNotFoundError(f"El directorio de corpus no existe: {corpus}")

    reader = SimpleDirectoryReader(
        input_dir=str(corpus),
        recursive=True,
        required_exts=SUPPORTED_EXTS,
        filename_as_id=True,
    )
    documents = reader.load_data()

    if not documents:
        raise ValueError(
            f"No se encontraron documentos soportados en {corpus} "
            f"(extensiones: {', '.join(SUPPORTED_EXTS)})."
        )

    # Normalizamos un metadato `source` estable (ruta relativa al corpus).
    for doc in documents:
        file_path = doc.metadata.get("file_path") or doc.metadata.get("file_name", "")
        try:
            source = str(Path(file_path).relative_to(corpus))
        except (ValueError, TypeError):
            source = Path(file_path).name or doc.doc_id
        doc.metadata["source"] = source

    return documents
