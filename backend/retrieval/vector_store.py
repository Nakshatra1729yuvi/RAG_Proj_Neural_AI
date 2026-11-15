import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from backend.ingestion.chunker import DocumentChunk

class FaissVectorStore:
    """
    FAISS-based vector index with metadata persistence and cosine similarity search.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product on normalized vectors = Cosine Similarity
        self.chunks: List[DocumentChunk] = []

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        if len(chunks) == 0:
            return
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")
        
        # Ensure float32 contiguous array
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(emb)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if self.index.ntotal == 0 or len(self.chunks) == 0:
            return []

        # Ensure query is 2D array
        q = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results: List[Tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def save(self, index_path: Path, metadata_path: Path):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        
        metadata = [chunk.model_dump() for chunk in self.chunks]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load(self, index_path: Path, metadata_path: Path) -> bool:
        if not (index_path.exists() and metadata_path.exists()):
            return False

        try:
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = [DocumentChunk(**item) for item in data]
            return True
        except Exception as e:
            print(f"Error loading FAISS store: {e}")
            self.clear()
            return False

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    def count(self) -> int:
        return self.index.ntotal

    def get_all_chunks(self) -> List[DocumentChunk]:
        return list(self.chunks)
