"""
Configuration for the RAG system (application layer).

Includes settings for:
- Models (embedding, reranker, LLM, vision)
- Retrieval parameters and thresholds
- Entity and intent system
- Disambiguation behaviour
"""
import os
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- PATHS ---
    DATA_DIR: str = "data"
    DB_PATH: str = "storage/qdrant_db"
    COLLECTION_NAME: str = "knowledge_base"

    # --- MODELS ---
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    LLM_MODEL: str = "qwen3:14b"
    # Vision model used for screenshot description (must be pulled in Ollama)
    VISION_MODEL: str = "qwen3-vl:8b"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- RETRIEVAL PARAMETERS ---
    TOP_K_RETRIEVAL: int = 80
    TOP_K_RERANK: int = 5
    # How many chunks to send into the disambiguator when building a list of documents
    TOP_K_FOR_DISAMBIG: int = 25
    
    SPARSE_MODEL: str = "cross-en-bge-m3"
    
    # Cosine‑similarity threshold before re‑ranking.
    # Filters out weak results early to reduce reranker load.
    COSINE_THRESHOLD: float = 0.3
    
    # Threshold for BGE‑Reranker logits.
    # For BGE‑Reranker: > 0 is very confident, > -2 ~80% confidence, > -4 is “possibly relevant”.
    SCORE_THRESHOLD: float = -2.0  # ~80% confidence — above this score we do not ask for clarification

    # --- ENTITY REGISTRY ---
    # Similarity threshold for entity clustering
    ENTITY_CLUSTER_THRESHOLD: float = 0.85
    # Minimal frequency for a “reliable” entity
    ENTITY_MIN_FREQUENCY: int = 3
    
    # --- INTENT CLASSIFIER ---
    # Threshold for ambiguous intent detection
    INTENT_AMBIGUITY_THRESHOLD: float = 0.3
    
    # --- DISAMBIGUATION ---
    # Minimal number of words for a query to be treated as specific
    MIN_QUERY_WORDS: int = 2
    # For queries with at least N words we skip document‑list clarifications and answer from all top chunks
    MIN_WORDS_SKIP_MULTIPLE_TOPICS: int = 3
    # Max number of options in clarification questions
    MAX_DISAMBIGUATION_OPTIONS: int = 5
    # Minimal score gap between top‑1 and top‑2 entities to treat as unambiguous
    ENTITY_AMBIGUITY_GAP: float = 0.1
    # Strict threshold for documents included into topic disambiguation lists
    # BGE‑Reranker: >= 0 — high confidence, show only such documents
    DISAMBIGUATION_SCORE_THRESHOLD: float = 0.0

    # --- PROMPTS ---
    SYSTEM_PROMPT: str = (
        "You are a senior assistant for internal documentation of a business application. "
        "Your task is to answer questions using ONLY the provided context.\n\n"
        "RULES:\n"
        "1. The context may contain screenshot descriptions (marked as [SCREENSHOT]); use them to provide precise navigation instructions (button names, menu paths).\n"
        "2. When the question is about a step‑by‑step procedure, quote UI element names exactly as they appear in the context.\n"
        "3. Always mention the source document name.\n"
        "4. If there is no relevant information in the context, answer: 'No relevant information was found in the knowledge base.'\n"
        "5. Do NOT invent information and do NOT rely on prior knowledge outside the context."
    )
    
    # Strict prompt for extraction‑only behaviour (no stylistic changes)
    STRICT_EXTRACTION_PROMPT: str = (
        "Extract ONLY the exact information from the context. "
        "Do NOT add anything from yourself. "
        "Quote verbatim where appropriate. "
        "Always mention the source."
    )


settings = Config()
