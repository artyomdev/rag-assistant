# Production RAG System

Production-style Retrieval-Augmented Generation (RAG) backend for question answering over document knowledge bases.

Built as an end-to-end system with retrieval, reranking, and GPU-based LLM inference.

---

## Key Features

- FastAPI backend API
- Vector search using Qdrant
- Local LLM inference via Ollama
- Embeddings + reranking pipeline (PyTorch / sentence-transformers)
- Document ingestion pipeline (PDF, Office, markdown)
- Docker / Docker Compose deployment
- Evaluation pipeline for answer quality

---

## Architecture (High-Level)

documents → chunking → embeddings → vector search → reranking → LLM → answer

---

## Tech Stack

**Backend:** Python, FastAPI  
**AI / ML:** RAG, embeddings, PyTorch, Transformers  
**Infra:** Docker, Linux  
**Storage:** Qdrant, MinIO  

---

## Why this project

- End-to-end RAG system (not a demo script)
- Backend + infrastructure focused (not UI-heavy)
- Uses local LLMs (no external API dependency)
- Designed for real-world workloads

---

## Quick Start

```bash
docker compose up --build



This repository contains a simplified and sanitized version of an internal RAG system, adapted for demonstration and educational purposes.

