from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel
from backend.ingestion.chunker import DocumentChunk
from backend.retrieval.embeddings import EmbeddingManager
from backend.retrieval.vector_store import FaissVectorStore
from backend.retrieval.bm25_retriever import BM25Index

class ScoredChunk(BaseModel):
    chunk: DocumentChunk
    rrf_score: float
    vector_rank: Optional[int] = None
    vector_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None

class HybridRetriever:
    """
    Hybrid retriever combining dense semantic search (FAISS) and sparse keyword search (BM25)
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: FaissVectorStore,
        bm25_index: BM25Index,
        embedding_manager: EmbeddingManager,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_manager = embedding_manager
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k_vector: int = 10,
        top_k_bm25: int = 10,
        top_k_final: int = 10
    ) -> List[ScoredChunk]:
        # 1. Vector Search
        query_emb = self.embedding_manager.embed_query(query)
        vector_results = self.vector_store.search(query_emb, top_k=top_k_vector)

        # 2. BM25 Search
        bm25_results = self.bm25_index.search(query, top_k=top_k_bm25)

        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}
        vector_meta: Dict[str, Tuple[int, float]] = {}
        bm25_meta: Dict[str, Tuple[int, float]] = {}

        # Process vector ranks
        for rank, (chunk, score) in enumerate(vector_results, start=1):
            chunk_id = chunk.chunk_id
            chunk_map[chunk_id] = chunk
            vector_meta[chunk_id] = (rank, score)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        # Process BM25 ranks
        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            chunk_id = chunk.chunk_id
            chunk_map[chunk_id] = chunk
            bm25_meta[chunk_id] = (rank, score)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        # Sort descending by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
        top_ids = sorted_ids[:min(top_k_final, len(sorted_ids))]

        results: List[ScoredChunk] = []
        for cid in top_ids:
            chunk = chunk_map[cid]
            v_rank, v_score = vector_meta.get(cid, (None, None))
            b_rank, b_score = bm25_meta.get(cid, (None, None))
            results.append(ScoredChunk(
                chunk=chunk,
                rrf_score=fused_scores[cid],
                vector_rank=v_rank,
                vector_score=v_score,
                bm25_rank=b_rank,
                bm25_score=b_score
            ))

        return results
