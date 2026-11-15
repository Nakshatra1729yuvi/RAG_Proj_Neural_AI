import pytest
from backend.generation.safety import SafetyGuard
from backend.generation.citation import CitationVerifier
from backend.ingestion.chunker import DocumentChunk

def test_emergency_detection():
    query_emergency = "I have sudden severe chest pain and cannot breathe!"
    is_em, disclaimer = SafetyGuard.check_emergency(query_emergency)
    assert is_em is True
    assert disclaimer is not None
    assert "911" in disclaimer

    query_normal = "What lifestyle changes help prevent high blood pressure?"
    is_em, disclaimer = SafetyGuard.check_emergency(query_normal)
    assert is_em is False
    assert disclaimer is None

def test_citation_verification():
    test_chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            doc_id="guideline",
            filename="guideline.pdf",
            page_number=3,
            chunk_index=0,
            text="Common blood pressure medications include ACE inhibitors and calcium channel blockers.",
            char_length=80,
            word_count=11
        )
    ]

    # Valid citation with grounding
    response_text = "Patients are commonly prescribed ACE inhibitors [Doc: guideline.pdf, p.3] to treat high blood pressure."
    report = CitationVerifier.verify(response_text, test_chunks)
    assert report.total_citations == 1
    assert report.valid_citations == 1
    assert report.grounded_citations == 1
    assert report.citation_accuracy == 100.0

    # Citation with hallucinated page
    bad_page_text = "Treatment includes ACE inhibitors [Doc: guideline.pdf, p.99]."
    bad_report = CitationVerifier.verify(bad_page_text, test_chunks)
    assert bad_report.total_citations == 1
    assert bad_report.citations[0].is_valid_source is True
    assert bad_report.citations[0].is_page_matched is False

    # Citation with completely made up document
    fake_doc_text = "Take 500mg daily [Doc: nonexistent_book.pdf, p.12]."
    fake_report = CitationVerifier.verify(fake_doc_text, test_chunks)
    assert fake_report.total_citations == 1
    assert fake_report.valid_citations == 0
