# Conceptos de RAG y su evaluación

## Qué es RAG

Retrieval-Augmented Generation (RAG) es una técnica que combina recuperación de información con generación de texto. En lugar de depender solo del conocimiento paramétrico del modelo, se recupera contexto relevante de un corpus externo y se lo entrega al LLM para que genere una respuesta fundamentada. Esto reduce las alucinaciones y permite responder sobre datos privados o actualizados.

## Chunking

El chunking es el proceso de dividir documentos en fragmentos (chunks) manejables. El tamaño del chunk y el solape (overlap) impactan la calidad: fragmentos muy grandes diluyen la relevancia, y muy chicos pierden contexto. Existen estrategias por oración (sentence splitting) y estrategias semánticas que agrupan por similitud.

## Embeddings y vector store

Un modelo de embeddings convierte cada fragmento en un vector numérico que captura su significado. Esos vectores se guardan en un vector store como Chroma o Qdrant, que permite búsqueda por similitud. Ante una consulta, se embebe la pregunta y se recuperan los fragmentos más cercanos en el espacio vectorial.

## Métricas de evaluación

La calidad de un RAG se mide, no se asume. Las métricas principales son:

- Faithfulness (fidelidad): mide si la respuesta se sostiene en el contexto recuperado, penalizando afirmaciones inventadas.
- Answer relevancy (relevancia de la respuesta): mide qué tan bien responde a la pregunta formulada.
- Context precision: mide si el contexto recuperado es efectivamente relevante.
- Context recall: mide si se recuperó toda la información necesaria para responder.

La herramienta RAGAS implementa estas métricas y permite comparar configuraciones distintas del pipeline, por ejemplo distintos tamaños de chunk o con y sin reranker, para decidir con datos cuál conviene.
