"""
Shared Embedder – a single shared embedding model instance.

Sharing one model instance across components (SearchEngine, EntityRegistry, etc.)
avoids duplicated VRAM usage.
"""
import logging
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

# Singleton instance
_embedder_instance: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """
    Return a singleton instance of the embedding model.
    
    The first call loads the model; subsequent calls reuse it.
    """
    global _embedder_instance
    
    if _embedder_instance is None:
        logger.info(f"🚀 Loading shared embedding model {settings.EMBEDDING_MODEL}")
        _embedder_instance = SentenceTransformer(
            settings.EMBEDDING_MODEL, 
            device=settings.DEVICE
        )
        logger.info(f"✅ Shared embedder loaded (dim={_embedder_instance.get_sentence_embedding_dimension()})")
    
    return _embedder_instance


def get_embedding_dimension() -> int:
    """Return embedding dimensionality."""
    return get_embedder().get_sentence_embedding_dimension()
