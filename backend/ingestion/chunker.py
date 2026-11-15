import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from backend.ingestion.extractor import DocumentPage

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    section_heading: Optional[str] = None
    char_length: int
    word_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StructureAwareChunker:
    """
    Structure-aware chunker that splits document pages into coherent chunks.
    Preserves paragraph and sentence boundaries, avoiding abrupt cuts.
    Retains page number, section headings, and origin metadata on every chunk.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        all_chunks = []
        for page in pages:
            page_chunks = self.chunk_single_page(page)
            all_chunks.extend(page_chunks)
        return all_chunks

    def chunk_single_page(self, page: DocumentPage) -> List[DocumentChunk]:
        text = page.text.strip()
        if not text:
            return []

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        # Build sentences
        sentence_units = []
        for para in paragraphs:
            # Check if this paragraph looks like a heading
            is_heading = len(para) < 80 and not para.endswith((".", ":", ";"))
            
            # Split into sentences using punctuation boundaries
            raw_sentences = re.split(r'(?<=[.!?])\s+', para)
            for s in raw_sentences:
                clean_s = s.strip()
                if clean_s:
                    sentence_units.append({
                        "text": clean_s,
                        "is_heading": is_heading,
                        "heading_text": para if is_heading else page.section_heading
                    })

        chunks: List[DocumentChunk] = []
        current_sentences: List[str] = []
        current_len = 0
        current_heading = page.section_heading
        chunk_idx = 0

        for unit in sentence_units:
            s_text = unit["text"]
            if unit["is_heading"]:
                current_heading = unit["heading_text"]

            # If adding this sentence exceeds chunk_size and we already have content
            if current_len + len(s_text) > self.chunk_size and current_sentences:
                chunk_str = " ".join(current_sentences).strip()
                if chunk_str:
                    c_id = f"{page.doc_id}_p{page.page_number}_c{chunk_idx}"
                    chunks.append(DocumentChunk(
                        chunk_id=c_id,
                        doc_id=page.doc_id,
                        filename=page.filename,
                        page_number=page.page_number,
                        chunk_index=chunk_idx,
                        text=chunk_str,
                        section_heading=current_heading,
                        char_length=len(chunk_str),
                        word_count=len(chunk_str.split()),
                        metadata={"total_pages": page.total_pages, "file_type": page.file_type}
                    ))
                    chunk_idx += 1

                # Overlap logic: keep trailing sentences up to chunk_overlap characters
                overlap_sentences = []
                overlap_len = 0
                for prev_s in reversed(current_sentences):
                    if overlap_len + len(prev_s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_s)
                        overlap_len += len(prev_s)
                    else:
                        break
                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(s_text)
            current_len += len(s_text)

        # Remaining sentences
        if current_sentences:
            chunk_str = " ".join(current_sentences).strip()
            if chunk_str:
                c_id = f"{page.doc_id}_p{page.page_number}_c{chunk_idx}"
                chunks.append(DocumentChunk(
                    chunk_id=c_id,
                    doc_id=page.doc_id,
                    filename=page.filename,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                    text=chunk_str,
                    section_heading=current_heading,
                    char_length=len(chunk_str),
                    word_count=len(chunk_str.split()),
                    metadata={"total_pages": page.total_pages, "file_type": page.file_type}
                ))

        return chunks
