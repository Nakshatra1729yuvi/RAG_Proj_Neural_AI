import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import settings
from backend.ingestion.chunker import StructureAwareChunker, DocumentChunk
from backend.retrieval.embeddings import EmbeddingManager
from backend.retrieval.vector_store import FaissVectorStore
from backend.retrieval.bm25_retriever import BM25Index
from backend.retrieval.hybrid_retriever import HybridRetriever, ScoredChunk
from backend.retrieval.reranker import CrossEncoderReranker, RerankedChunk
from backend.storage.document_store import DocumentStore, DocumentRecord
from backend.generation.llm_client import OllamaClient
from backend.generation.generator import AnswerGenerator, GenerationResult

logger = logging.getLogger(__name__)

class MediRAGPipeline:
    """
    End-to-end MediRAG orchestration pipeline.
    Combines structure-aware extraction, FAISS vector search, BM25 keyword retrieval,
    RRF hybrid fusion, Cross-Encoder reranking, and grounded LLM generation with citation auditing.
    """

    _instance = None

    def __init__(self):
        logger.info("Initializing MediRAG Pipeline components...")
        self.chunker = StructureAwareChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.embedding_manager = EmbeddingManager.get_instance()
        self.vector_store = FaissVectorStore(dimension=self.embedding_manager.dimension)
        self.bm25_index = BM25Index()

        self.document_store = DocumentStore(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            embedding_manager=self.embedding_manager,
            chunker=self.chunker,
            storage_dir=settings.INDICES_DIR
        )

        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            embedding_manager=self.embedding_manager,
            rrf_k=settings.RRF_K
        )

        self.reranker = CrossEncoderReranker.get_instance()
        self.llm_client = OllamaClient()
        self.generator = AnswerGenerator(llm_client=self.llm_client)
        logger.info("MediRAG Pipeline initialized successfully.")

    @classmethod
    def get_instance(cls) -> "MediRAGPipeline":
        if cls._instance is None:
            cls._instance = MediRAGPipeline()
        return cls._instance

    def query(
        self,
        question: str,
        search_mode: str = "hybrid_rerank",
        top_k: int = settings.TOP_K_FINAL,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Executes query through specified retrieval path and generates grounded answer.
        Modes: 'vector', 'bm25', 'hybrid', 'hybrid_rerank'
        """
        if self.document_store.total_chunks() == 0:
            return {
                "answer": "No medical documents have been indexed yet. Please upload medical reference documents first.",
                "raw_answer": "",
                "is_emergency": False,
                "is_supported": False,
                "retrieval_mode": search_mode,
                "citations": [],
                "citation_report": {},
                "evidence_chunks": [],
                "follow_up_questions": []
            }

        retrieved_chunks: List[DocumentChunk] = []
        scored_evidence: List[Dict[str, Any]] = []
        min_relevance_passed = True

        if search_mode == "vector":
            query_emb = self.embedding_manager.embed_query(question)
            results = self.vector_store.search(query_emb, top_k=top_k)
            retrieved_chunks = [c for c, _ in results]
            for c, score in results:
                scored_evidence.append({
                    "chunk": c.model_dump(),
                    "score": round(float(score), 4),
                    "score_type": "cosine_similarity"
                })
            # Cosine similarity threshold for vector only
            if results and results[0][1] < 0.25:
                min_relevance_passed = False

        elif search_mode == "bm25":
            results = self.bm25_index.search(question, top_k=top_k)
            retrieved_chunks = [c for c, _ in results]
            for c, score in results:
                scored_evidence.append({
                    "chunk": c.model_dump(),
                    "score": round(float(score), 4),
                    "score_type": "bm25_score"
                })
            if not results:
                min_relevance_passed = False

        elif search_mode == "hybrid":
            hybrid_results: List[ScoredChunk] = self.hybrid_retriever.retrieve(
                query=question,
                top_k_vector=settings.TOP_K_VECTOR,
                top_k_bm25=settings.TOP_K_BM25,
                top_k_final=top_k
            )
            retrieved_chunks = [item.chunk for item in hybrid_results]
            for item in hybrid_results:
                scored_evidence.append({
                    "chunk": item.chunk.model_dump(),
                    "score": round(float(item.rrf_score), 4),
                    "score_type": "rrf_score",
                    "vector_rank": item.vector_rank,
                    "bm25_rank": item.bm25_rank
                })
            if not hybrid_results:
                min_relevance_passed = False

        elif search_mode == "hybrid_rerank":
            # 1. Hybrid candidate generation
            hybrid_candidates = self.hybrid_retriever.retrieve(
                query=question,
                top_k_vector=settings.TOP_K_VECTOR,
                top_k_bm25=settings.TOP_K_BM25,
                top_k_final=settings.TOP_K_RRF
            )

            # 2. Cross-Encoder reranking
            reranked: List[RerankedChunk] = self.reranker.rerank(
                query=question,
                candidates=hybrid_candidates,
                top_k=top_k
            )

            retrieved_chunks = [item.chunk for item in reranked]
            for item in reranked:
                scored_evidence.append({
                    "chunk": item.chunk.model_dump(),
                    "score": round(float(item.rerank_score), 4),
                    "score_type": "cross_encoder_logit",
                    "rrf_score": item.rrf_score,
                    "vector_rank": item.vector_rank,
                    "bm25_rank": item.bm25_rank
                })

            # Check cross-encoder minimum threshold
            if not reranked or reranked[0].rerank_score < settings.MIN_RELEVANCE_SCORE:
                min_relevance_passed = False

        else:
            raise ValueError(f"Unknown search mode: {search_mode}")

        # Generate answer
        gen_result: GenerationResult = self.generator.generate_answer(
            query=question,
            retrieved_chunks=retrieved_chunks,
            conversation_history=history,
            min_relevance_passed=min_relevance_passed
        )

        return {
            "answer": gen_result.answer,
            "raw_answer": gen_result.raw_answer,
            "is_emergency": gen_result.is_emergency,
            "is_supported": gen_result.is_supported,
            "retrieval_mode": search_mode,
            "citations": [c.model_dump() for c in gen_result.citation_report.citations],
            "citation_report": gen_result.citation_report.model_dump(),
            "evidence_chunks": scored_evidence,
            "follow_up_questions": gen_result.follow_up_questions
        }

    def ingest_file(self, file_path: Path) -> DocumentRecord:
        return self.document_store.add_document(file_path)

    def delete_file(self, doc_id: str) -> bool:
        return self.document_store.delete_document(doc_id)

    def get_documents(self) -> List[DocumentRecord]:
        return self.document_store.get_documents()

    def get_stats(self) -> Dict[str, Any]:
        docs = self.document_store.get_documents()
        total_pages = sum(d.num_pages for d in docs)
        return {
            "total_documents": len(docs),
            "total_chunks": self.document_store.total_chunks(),
            "total_pages": total_pages,
            "llm_model": self.llm_client.active_model,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "device": settings.DEVICE
        }
