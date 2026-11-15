import pytest
import numpy as np
from backend.retrieval.embeddings import EmbeddingManager
from backend.retrieval.vector_store import FaissVectorStore
from backend.retrieval.bm25_retriever import BM25Index
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.ingestion.chunker import DocumentChunk

@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            doc_id="doc_hyper",
            filename="hyper.pdf",
            page_number=1,
            chunk_index=0,
            text="Hypertension is defined as blood pressure of 140/90 mmHg or higher.",
            section_heading="Diagnosis",
            char_length=68,
            word_count=10
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            doc_id="doc_diab",
            filename="diab.docx",
            page_number=2,
            chunk_index=0,
            text="Type 2 diabetes affects how the body uses insulin and glucose.",
            section_heading="Types",
            char_length=62,
            word_count=10
        )
    ]

def test_embeddings_generation():
    emb_mgr = EmbeddingManager.get_instance()
    texts = ["High blood pressure treatment", "Insulin resistance"]
    vectors = emb_mgr.embed_texts(texts)
    assert vectors.shape == (2, emb_mgr.dimension)
    # Check normalization (norm should be ~1.0)
    norm = np.linalg.norm(vectors[0])
    assert np.isclose(norm, 1.0, atol=1e-3)

def test_faiss_vector_store(sample_chunks):
    emb_mgr = EmbeddingManager.get_instance()
    store = FaissVectorStore(dimension=emb_mgr.dimension)
    vectors = emb_mgr.embed_texts([c.text for c in sample_chunks])
    store.add_chunks(sample_chunks, vectors)

    assert store.count() == 2
    q_vec = emb_mgr.embed_query("blood pressure reading")
    results = store.search(q_vec, top_k=2)
    assert len(results) == 2
    assert results[0][0].chunk_id == "chunk_1"

def test_bm25_retriever(sample_chunks):
    bm25 = BM25Index()
    bm25.build_index(sample_chunks)
    assert bm25.count() == 2

    results = bm25.search("insulin glucose", top_k=2)
    assert len(results) >= 1
    assert results[0][0].chunk_id == "chunk_2"

def test_hybrid_retrieval(sample_chunks):
    emb_mgr = EmbeddingManager.get_instance()
    store = FaissVectorStore(dimension=emb_mgr.dimension)
    vectors = emb_mgr.embed_texts([c.text for c in sample_chunks])
    store.add_chunks(sample_chunks, vectors)

    bm25 = BM25Index()
    bm25.build_index(sample_chunks)

    hybrid = HybridRetriever(store, bm25, emb_mgr, rrf_k=60)
    results = hybrid.retrieve("blood pressure diagnosis", top_k_final=2)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "chunk_1"

def test_cross_encoder_reranker(sample_chunks):
    reranker = CrossEncoderReranker.get_instance()
    results = reranker.rerank("What causes diabetes?", sample_chunks, top_k=2)
    assert len(results) == 2
    # The chunk talking about diabetes should score higher
    assert results[0].chunk.chunk_id == "chunk_2"
