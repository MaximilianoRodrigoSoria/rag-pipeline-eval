# Portfolio AI Engineer — Visión general

Este portfolio reúne cinco proyectos que demuestran competencias de AI Engineer / AI Architect: RAG con evaluación, orquestación multi-agente, un servidor MCP, observabilidad LLMOps y arquitectura cloud para GenAI.

## rag-pipeline-eval

Es el proyecto base. Construye un pipeline de Retrieval-Augmented Generation sobre un corpus propio y agrega un módulo de evaluación automatizada. La ingesta carga documentos, el chunking los trocea en fragmentos, un modelo de embeddings los convierte en vectores y un vector store (Chroma) los almacena. Ante una pregunta, el sistema recupera los fragmentos más relevantes y un LLM genera la respuesta fundamentada. El diferenciador es medir la calidad con métricas como faithfulness (fidelidad al contexto) y answer relevancy (relevancia de la respuesta).

## multi-agent-orchestration

Un sistema con varios agentes especializados (planner, researcher, executor, critic) coordinados por un grafo de estados. Incluye un paso de human-in-the-loop: antes de ejecutar una acción crítica, el flujo se detiene y espera aprobación humana. El estado se persiste para poder pausar y reanudar.

## mcp-server-demo

Un servidor que implementa el Model Context Protocol y expone herramientas propias (consultas a una base SQLite y a una API externa) para que un LLM las use de forma estandarizada. Funciona por stdio en local y, opcionalmente, por HTTP remoto.

## llmops-observability

Una capa de observabilidad que loguea latencia, costo por request y tasa de error de un servicio LLM. Se containeriza con Docker y se despliega con un pipeline de CI/CD en GitHub Actions.

## genai-cloud-architecture

Un diagrama de arquitectura cloud escalable y multi-tenant para servir modelos LLM en AWS o GCP, con una implementación mínima de referencia que demuestra aislamiento por tenant, rate limiting y caching semántico.
