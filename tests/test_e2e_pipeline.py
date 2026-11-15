import pytest
from backend.pipeline import MediRAGPipeline

@pytest.fixture(scope="module")
def pipeline():
    return MediRAGPipeline.get_instance()

def test_pipeline_stats(pipeline):
    stats = pipeline.get_stats()
    assert stats["total_documents"] >= 4
    assert stats["total_chunks"] >= 100

@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid", "hybrid_rerank"])
def test_retrieval_modes(pipeline, mode):
    res = pipeline.query(
        question="What are the symptoms of high blood pressure?",
        search_mode=mode,
        top_k=3
    )
    assert "answer" in res
    assert len(res["evidence_chunks"]) >= 1
    assert res["retrieval_mode"] == mode

def test_emergency_query_handling(pipeline):
    res = pipeline.query(
        question="I am having crushing chest pain radiating to left arm and shortness of breath",
        search_mode="hybrid_rerank",
        top_k=3
    )
    assert res["is_emergency"] is True
    assert "EMERGENCY NOTICE" in res["answer"]
    assert "911" in res["answer"]

def test_unsupported_query_handling(pipeline):
    res = pipeline.query(
        question="What is the surgical bypass incision technique for intracranial berry aneurysm clipping?",
        search_mode="hybrid_rerank",
        top_k=3
    )
    # The pipeline should either indicate unsupported or lack of evidence
    assert ("insufficient evidence" in res["answer"].lower() or 
            "not provide" in res["answer"].lower() or 
            res["is_supported"] is False)
