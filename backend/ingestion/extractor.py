import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DocumentPage(BaseModel):
    doc_id: str
    filename: str
    page_number: int  # 1-indexed
    total_pages: int
    text: str
    section_heading: Optional[str] = None
    file_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentExtractor:
    """
    Extracts text and page-level metadata from PDF, DOCX, and TXT files.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    @classmethod
    def extract(cls, file_path: str | Path, doc_id: Optional[str] = None) -> List[DocumentPage]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{ext}'. Supported: {cls.SUPPORTED_EXTENSIONS}")

        doc_identifier = doc_id or path.stem

        if ext == ".pdf":
            return cls._extract_pdf(path, doc_identifier)
        elif ext == ".docx":
            return cls._extract_docx(path, doc_identifier)
        elif ext == ".txt":
            return cls._extract_txt(path, doc_identifier)
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    @classmethod
    def _extract_pdf(cls, path: Path, doc_id: str) -> List[DocumentPage]:
        import pymupdf as fitz
        pages = []
        doc = fitz.open(str(path))
        total_pages = len(doc)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_text = page.get_text("text").strip()
            
            # Clean non-printable characters while preserving newlines
            cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', page_text)
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

            # Heuristic to detect top section heading
            heading = None
            lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
            if lines:
                first_line = lines[0]
                if len(first_line) < 80 and not first_line.endswith("."):
                    heading = first_line

            pages.append(DocumentPage(
                doc_id=doc_id,
                filename=path.name,
                page_number=page_idx + 1,
                total_pages=total_pages,
                text=cleaned_text,
                section_heading=heading,
                file_type="pdf",
                metadata={"file_size": path.stat().st_size}
            ))

        doc.close()
        return pages

    @classmethod
    def _extract_docx(cls, path: Path, doc_id: str) -> List[DocumentPage]:
        import docx
        doc = docx.Document(str(path))
        
        # In docx, documents don't have rigid physical page numbers.
        # We partition based on headings or chunk blocks to create logical pages.
        logical_pages = []
        current_page_lines = []
        current_heading = None
        current_char_count = 0
        approx_page_size = 2000  # ~350-400 words per logical page

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Check if paragraph is a heading
            if para.style.name.startswith("Heading"):
                if current_page_lines:
                    page_text = "\n\n".join(current_page_lines)
                    logical_pages.append((current_heading, page_text))
                    current_page_lines = []
                    current_char_count = 0
                current_heading = text

            current_page_lines.append(text)
            current_char_count += len(text)

            if current_char_count >= approx_page_size:
                page_text = "\n\n".join(current_page_lines)
                logical_pages.append((current_heading, page_text))
                current_page_lines = []
                current_char_count = 0

        if current_page_lines:
            page_text = "\n\n".join(current_page_lines)
            logical_pages.append((current_heading, page_text))

        if not logical_pages:
            logical_pages.append((None, ""))

        total_pages = len(logical_pages)
        pages = []
        for idx, (heading, text) in enumerate(logical_pages):
            pages.append(DocumentPage(
                doc_id=doc_id,
                filename=path.name,
                page_number=idx + 1,
                total_pages=total_pages,
                text=text,
                section_heading=heading,
                file_type="docx",
                metadata={"file_size": path.stat().st_size}
            ))
        return pages

    @classmethod
    def _extract_txt(cls, path: Path, doc_id: str) -> List[DocumentPage]:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")

        # Partition text into ~2000 character logical pages preserving paragraph boundaries
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        pages = []
        current_lines = []
        current_size = 0
        logical_page_texts = []

        for p in paragraphs:
            current_lines.append(p)
            current_size += len(p)
            if current_size >= 2000:
                logical_page_texts.append("\n\n".join(current_lines))
                current_lines = []
                current_size = 0

        if current_lines:
            logical_page_texts.append("\n\n".join(current_lines))

        if not logical_page_texts:
            logical_page_texts.append("")

        total_pages = len(logical_page_texts)
        for idx, text in enumerate(logical_page_texts):
            # Extract heading if available
            first_line = text.splitlines()[0].strip() if text.strip() else None
            heading = first_line if (first_line and len(first_line) < 80) else None
            pages.append(DocumentPage(
                doc_id=doc_id,
                filename=path.name,
                page_number=idx + 1,
                total_pages=total_pages,
                text=text,
                section_heading=heading,
                file_type="txt",
                metadata={"file_size": path.stat().st_size}
            ))

        return pages
