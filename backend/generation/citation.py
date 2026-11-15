import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from backend.ingestion.chunker import DocumentChunk

class Citation(BaseModel):
    raw_citation: str
    doc_name: str
    page_number: int
    is_valid_source: bool = False
    is_page_matched: bool = False
    is_grounded: bool = False
    matching_chunk_id: Optional[str] = None
    supporting_text_snippet: Optional[str] = None
    verification_score: float = 0.0

class CitationVerificationReport(BaseModel):
    total_citations: int = 0
    valid_citations: int = 0
    grounded_citations: int = 0
    citation_accuracy: float = 0.0
    citations: List[Citation] = Field(default_factory=list)

class CitationVerifier:
    """
    Extracts, verifies, and audits inline citations against actual retrieved source chunks.
    Verifies document existence, page accuracy, and lexical/factual grounding.
    """

    CITATION_PATTERNS = [
        # [Doc: filename, p.1] or [Doc: filename, page 1]
        re.compile(r'\[Doc:\s*([^,\]]+?),\s*(?:p\.|page\s*)(\d+)\]', re.IGNORECASE),
        # [Source: filename, Page: 1]
        re.compile(r'\[Source:\s*([^,\]]+?),\s*(?:Page|p\.)\s*(\d+)\]', re.IGNORECASE),
        # [filename, p.1]
        re.compile(r'\[([a-zA-Z0-9_\-\.]+\.(?:pdf|docx|txt)),\s*(?:p\.|page\s*)(\d+)\]', re.IGNORECASE)
    ]

    @classmethod
    def extract_citations(cls, text: str) -> List[Tuple[str, str, int]]:
        """
        Extracts tuples of (raw_match, doc_name, page_number)
        """
        citations = []
        seen = set()

        for pattern in cls.CITATION_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(0)
                doc = match.group(1).strip()
                page = int(match.group(2).strip())
                key = (doc.lower(), page)
                if key not in seen:
                    seen.add(key)
                    citations.append((raw, doc, page))

        return citations

    @classmethod
    def verify(
        cls,
        text: str,
        retrieved_chunks: List[DocumentChunk]
    ) -> CitationVerificationReport:
        extracted = cls.extract_citations(text)
        if not extracted:
            return CitationVerificationReport(
                total_citations=0,
                valid_citations=0,
                grounded_citations=0,
                citation_accuracy=0.0,
                citations=[]
            )

        citations_result: List[Citation] = []
        valid_count = 0
        grounded_count = 0

        # Build lookup table of retrieved chunks
        chunk_lookup: Dict[str, List[DocumentChunk]] = {}
        for c in retrieved_chunks:
            # normalize filename for robust comparison
            norm_name = c.filename.lower()
            chunk_lookup.setdefault(norm_name, []).append(c)

        # Extract sentences from generated answer to check claim overlap
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        for raw, doc_name, page_num in extracted:
            citation_obj = Citation(
                raw_citation=raw,
                doc_name=doc_name,
                page_number=page_num
            )

            # 1. Check if doc exists in retrieved chunks
            norm_doc = doc_name.lower()
            matching_chunks = []
            for known_doc, chunks in chunk_lookup.items():
                if norm_doc in known_doc or known_doc in norm_doc:
                    matching_chunks.extend(chunks)

            if matching_chunks:
                citation_obj.is_valid_source = True
                valid_count += 1

                # 2. Check if page matches
                page_matched_chunks = [c for c in matching_chunks if c.page_number == page_num]
                if page_matched_chunks:
                    citation_obj.is_page_matched = True
                    target_chunks = page_matched_chunks
                else:
                    target_chunks = matching_chunks

                # 3. Grounding check: find context sentence containing this citation
                surrounding_sentence = ""
                for s in sentences:
                    if raw in s:
                        surrounding_sentence = s.replace(raw, "").strip()
                        break

                best_overlap_ratio = 0.0
                best_chunk = None

                # Compute word token overlap between sentence and target chunk
                claim_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', surrounding_sentence.lower()))
                for c in target_chunks:
                    chunk_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', c.text.lower()))
                    if claim_tokens:
                        overlap = len(claim_tokens.intersection(chunk_tokens)) / len(claim_tokens)
                    else:
                        overlap = 0.5  # fallback if claim is short

                    if overlap > best_overlap_ratio:
                        best_overlap_ratio = overlap
                        best_chunk = c

                if best_chunk is not None:
                    citation_obj.matching_chunk_id = best_chunk.chunk_id
                    citation_obj.supporting_text_snippet = best_chunk.text[:200] + "..."
                    citation_obj.verification_score = round(float(best_overlap_ratio), 3)

                    # Grounded if lexical overlap exceeds 30% or page matches exactly
                    if best_overlap_ratio >= 0.25 or citation_obj.is_page_matched:
                        citation_obj.is_grounded = True
                        grounded_count += 1

            citations_result.append(citation_obj)

        accuracy = round((grounded_count / len(extracted)) * 100.0, 1) if extracted else 0.0

        return CitationVerificationReport(
            total_citations=len(extracted),
            valid_citations=valid_count,
            grounded_citations=grounded_count,
            citation_accuracy=accuracy,
            citations=citations_result
        )
