import pytest
from pathlib import Path
from backend.ingestion.extractor import DocumentExtractor, DocumentPage
from backend.ingestion.chunker import StructureAwareChunker, DocumentChunk
from config import settings

def test_extract_pdf():
    pdf_path = settings.RAW_DIR / "who_hypertension_guideline.pdf"
    assert pdf_path.exists(), "Sample PDF must exist"
    pages = DocumentExtractor.extract(pdf_path)
    assert len(pages) > 0
    assert pages[0].page_number == 1
    assert pages[0].file_type == "pdf"
    assert len(pages[0].text) > 0

def test_extract_docx():
    docx_path = settings.RAW_DIR / "who_diabetes_factsheet.docx"
    assert docx_path.exists(), "Sample DOCX must exist"
    pages = DocumentExtractor.extract(docx_path)
    assert len(pages) > 0
    assert pages[0].file_type == "docx"
    assert len(pages[0].text) > 0

def test_extract_txt():
    txt_path = settings.RAW_DIR / "who_cardiovascular_diseases.txt"
    assert txt_path.exists(), "Sample TXT must exist"
    pages = DocumentExtractor.extract(txt_path)
    assert len(pages) > 0
    assert pages[0].file_type == "txt"
    assert len(pages[0].text) > 0

def test_structure_aware_chunker():
    dummy_page = DocumentPage(
        doc_id="test_doc",
        filename="test_doc.pdf",
        page_number=1,
        total_pages=1,
        text="Hypertension is high blood pressure. It affects over 1 billion people worldwide. Symptoms can include headaches and blurred vision. Salt intake should be reduced.",
        section_heading="Overview",
        file_type="pdf"
    )
    chunker = StructureAwareChunker(chunk_size=100, chunk_overlap=30)
    chunks = chunker.chunk_pages([dummy_page])
    assert len(chunks) >= 1
    assert all(c.page_number == 1 for c in chunks)
    assert all(c.filename == "test_doc.pdf" for c in chunks)
    assert all(c.section_heading == "Overview" for c in chunks)
