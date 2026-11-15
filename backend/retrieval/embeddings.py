import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import settings

class EmbeddingManager:
    """
    Handles dense embedding generation using SentenceTransformers.
    Generates normalized embeddings suitable for inner-product (cosine) search.
    """

    _instance = None

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL, device: str = settings.DEVICE):
        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name, device=device)
        if hasattr(self._model, "get_embedding_dimension"):
            self.dimension = self._model.get_embedding_dimension()
        else:
            self.dimension = self._model.get_sentence_embedding_dimension()

    @classmethod
    def get_instance(cls) -> "EmbeddingManager":
        if cls._instance is None:
            cls._instance = EmbeddingManager()
        return cls._instance

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self._model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.astype(np.float32)
