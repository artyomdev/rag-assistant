"""
SearchEngine – vector search with optional entity‑based filtering.

Features:
- Search by canonical query text.
- Optional filtering by entity_ids.
- Support for structured metadata.
"""
import os
import logging
import uuid
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchAny
)
from app.config import settings
from app.embedder import get_embedder, get_embedding_dimension

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self):
        # Initialise local Qdrant (embedded) database
        db_full_path = os.path.abspath(settings.DB_PATH)
        os.makedirs(os.path.dirname(db_full_path), exist_ok=True)
        self.client = QdrantClient(path=db_full_path)

        self.embedder = get_embedder()

        self.dimension = get_embedding_dimension()

        self._ensure_collection()

    def _ensure_collection(self):
        """Create the collection if it does not exist yet."""
        if not self.client.collection_exists("knowledge_base"):
            logger.info("📦 Creating collection knowledge_base")
            self.client.create_collection(
                collection_name="knowledge_base",
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
            )

    def index_documents(self, documents: list):
        """Index a list of documents into Qdrant.

        Each document is a dict with "text" and optional "metadata".
        """
        points = []
        logger.info(f"📌 Indexing {len(documents)} documents into Qdrant...")

        texts = [doc.get("text", "") for doc in documents]

        dense_vectors = self.embedder.encode(texts, normalize_embeddings=True)

        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {}) or {}
            point_id = str(uuid.uuid4())

            points.append(
                PointStruct(
                    id=point_id,
                    vector=dense_vectors[i].tolist(),
                    payload={"text": text, **metadata}
                )
            )

        batch_size = 100
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name="knowledge_base",
                points=points[start : start + batch_size]
            )
            logger.info(f"   Indexed {min(start+batch_size, len(points))}/{len(points)}")

        logger.info("✅ Indexing finished")

    def search(
        self, 
        query: str, 
        limit: int = 30,
        entity_ids: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for semantically similar documents.

        Args:
            query: user query text
            limit: maximum number of results
            entity_ids: optional entity filter (at least one id must match)
            category: optional category filter
        """
        # 1. Generate embedding for the query
        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()

        query_filter = None
        conditions = []
        
        if entity_ids:
            conditions.append(
                FieldCondition(
                    key="entity_ids",
                    match=MatchAny(any=entity_ids)
                )
            )
        
        if category:
            conditions.append(
                FieldCondition(
                    key="category",
                    match=MatchAny(any=[category])
                )
            )
        
        if conditions:
            query_filter = Filter(must=conditions)

        response = self.client.query_points(
            collection_name="knowledge_base",
            query=query_vector, 
            limit=limit,
            with_payload=True,
            query_filter=query_filter
        )

        results = []
        for hit in response.points:
            results.append(hit.payload)
            
        return results

    def search_with_scores(
        self,
        query: str,
        limit: int = 30,
        entity_ids: Optional[List[str]] = None
    ) -> List[tuple]:
        """Search and return both payload and scores (useful for debugging)."""
        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        
        query_filter = None
        if entity_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="entity_ids",
                        match=MatchAny(any=entity_ids)
                    )
                ]
            )

        response = self.client.query_points(
            collection_name="knowledge_base",
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter
        )

        results = []
        for hit in response.points:
            results.append((hit.payload, hit.score))
            
        return results

    def get_by_entity_id(self, entity_id: str, limit: int = 10) -> List[Dict]:
        """Return documents linked to a particular entity id."""
        response = self.client.scroll(
            collection_name="knowledge_base",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="entity_ids",
                        match=MatchAny(any=[entity_id])
                    )
                ]
            ),
            limit=limit,
            with_payload=True
        )
        
        results = []
        for point in response[0]:  # scroll returns (points, next_page_offset)
            results.append(point.payload)
        
        return results

    def close(self):
        """Close the underlying Qdrant client."""
        if hasattr(self, "client"):
            self.client.close()
