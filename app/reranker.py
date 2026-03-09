from sentence_transformers import CrossEncoder
from app.config import settings
import torch

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(
            settings.RERANKER_MODEL, 
            device=settings.DEVICE,
            model_kwargs={"torch_dtype": torch.float16 if settings.DEVICE == "cuda" else torch.float32}
        )

    def rank(self, query: str, documents: list, top_k: int = 5):
        if not documents:
            return []

        pairs = [[query, doc.get('text', '')] for doc in documents]
        
        scores = self.model.predict(pairs).tolist() 
        if isinstance(scores, float):
            scores = [scores]

        for i, doc in enumerate(documents):
            doc['score'] = float(scores[i])

        ranked = sorted(documents, key=lambda x: x['score'], reverse=True)
        return ranked[:top_k]