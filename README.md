# Production RAG System (Local LLM + GPU Inference)

End-to-end Retrieval-Augmented Generation system with local GPU-based inference and offline deployment.

Built to operate under real-world constraints: limited GPU memory, no external APIs, and production reliability requirements.

---

## Why this project

This system was designed for environments where:

- internet access is restricted (air-gapped infrastructure)
- GPU resources are limited
- inference cost and latency must be controlled

---

## Architecture

Pipeline:

document ingestion → embeddings → vector search → reranking → LLM inference

Core components:

- Embeddings: bge-m3  
- Vector DB: Qdrant (on-disk)  
- Reranking: CrossEncoder  
- LLM: local inference via Ollama  
- API: FastAPI  

---

## Key engineering decisions

- Reduced reranking cost via candidate filtering  
  (**80 → 25 → 5** instead of reranking full candidate set)

- Optimized GPU usage:
  - FP16 inference for reranker
  - shared embedding model (reduced duplicate VRAM usage)
  - reduced context size for inference

- Vision ingestion optimization:
  - image downscaling (1280 → 768)
  - JPEG compression
  - request throttling to avoid GPU overload

- Batched vector indexing (100 vectors per upsert)

---

## Real-world constraints

- Fully offline / air-gapped deployment  
- No runtime model downloads (all models pre-cached in Docker images)  
- GPU shared between embedding, reranking, and LLM inference  

---

## Production issues

- Power outage caused model cache loss  
  → fixed by embedding model weights into Docker images  

- GPU contention between services  
  → mitigated via sequential processing and throttling  

---

## Scale

- 2000+ documents (including UI screenshots)  
- 100k+ vectors stored in Qdrant  
- ~72h full ingestion pipeline runtime  

---

## Tech stack

Python • FastAPI • Docker • Qdrant • PyTorch • CUDA • Ollama  
Transformers • SentenceTransformers • RAG • Vector Search  

---

## Run locally

```bash
docker compose up --build

