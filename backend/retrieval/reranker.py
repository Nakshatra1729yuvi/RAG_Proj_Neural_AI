from typing import List, Tuple, Optional
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from config import settings
from backend.ingestion.chunker import DocumentChunk
from backend.retrieval.hybrid_retriever import ScoredChunk

class RerankedChunk(BaseModel):
    chunk: DocumentChunk
    rerank_score: float
    rrf_score: Optional[float] = None
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None

class CrossEncoderReranker:
    """
    Reranks candidate retrieved chunks using a cross-encoder model.
    Computes cross-attention between the query and candidate chunk for high-fidelity relevance scoring.
    """

    _instance = None

    def __init__(self, model_name: str = settings.RERANKER_MODEL, device: str = settings.DEVICE):
        self.model_name = model_name
        self.device = device
        self._model = CrossEncoder(model_name, device=device)

    @classmethod
    def get_instance(cls) -> "CrossEncoderReranker":
        if cls._instance is None:
            cls._instance = CrossEncoderReranker()
        return cls._instance

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk] | List[DocumentChunk],
        top_k: int = 4
    ) -> List[RerankedChunk]:
        if not candidates:
            return []

        # Extract chunks and any previous metadata
        chunks: List[DocumentChunk] = []
        rrf_map = {}
        v_map = {}
        b_map = {}

        for item in candidates:
            if isinstance(item, ScoredChunk):
                chunks.append(item.chunk)
                rrf_map[item.chunk.chunk_id] = item.rrf_score
                v_map[item.chunk.chunk_id] = item.vector_rank
                b_map[item.chunk.chunk_id] = item.bm25_rank
            else:
                chunks.append(item)

        # Cross-encoder pairs
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Pair up and rank
        scored_pairs = list(zip(chunks, scores))
        scored_pairs.sort(key=lambda x: float(x[1]), reverse=True)

        reranked_results: List[RerankedChunk] = []
        for chunk, score in scored_pairs[:top_k]:
            reranked_results.append(RerankedChunk(
                chunk=chunk,
                rerank_score=float(score),
                rrf_score=rrf_map.get(chunk.chunk_id),
                vector_rank=v_map.get(chunk.chunk_id),
                bm25_rank=b_map.get(chunk.chunk_id)
            ))

        return reranked_results
