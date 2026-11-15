import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from config import settings
from backend.ingestion.extractor import DocumentExtractor, DocumentPage
from backend.ingestion.chunker import StructureAwareChunker, DocumentChunk
from backend.retrieval.embeddings import EmbeddingManager
from backend.retrieval.vector_store import FaissVectorStore
from backend.retrieval.bm25_retriever import BM25Index

class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    file_size: int
    num_pages: int
    num_chunks: int
    uploaded_at: str
    status: str = "indexed"  # "indexed", "failed", "processing"

class DocumentStore:
    """
    Manages document persistence, chunk registry, and index synchronizations.
    """

    def __init__(
        self,
        vector_store: FaissVectorStore,
        bm25_index: BM25Index,
        embedding_manager: EmbeddingManager,
        chunker: StructureAwareChunker,
        storage_dir: Path = settings.INDICES_DIR
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_manager = embedding_manager
        self.chunker = chunker
        self.storage_dir = storage_dir
        self.registry_file = storage_dir / "documents_registry.json"
        self.documents: Dict[str, DocumentRecord] = {}

        self.faiss_index_path = storage_dir / "faiss.index"
        self.faiss_meta_path = storage_dir / "faiss_meta.json"
        self.bm25_path = storage_dir / "bm25.json"

        self.load_all()

    def load_all(self):
        # Load registry
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = {k: DocumentRecord(**v) for k, v in data.items()}
            except Exception as e:
                print(f"Error loading document registry: {e}")
                self.documents = {}

        # Load indices
        self.vector_store.load(self.faiss_index_path, self.faiss_meta_path)
        self.bm25_index.load(self.bm25_path)

    def save_all(self):
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # Save registry
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump({k: v.model_dump() for k, v in self.documents.items()}, f, indent=2)

        # Save indices
        self.vector_store.save(self.faiss_index_path, self.faiss_meta_path)
        self.bm25_index.save(self.bm25_path)

    def add_document(self, file_path: Path, doc_id: Optional[str] = None) -> DocumentRecord:
        identifier = doc_id or file_path.stem

        # 1. Extract pages
        pages: List[DocumentPage] = DocumentExtractor.extract(file_path, doc_id=identifier)
        if not pages:
            raise ValueError(f"No content extracted from {file_path.name}")

        # 2. Chunk pages
        chunks: List[DocumentChunk] = self.chunker.chunk_pages(pages)
        if not chunks:
            raise ValueError(f"No chunks created for {file_path.name}")

        # 3. Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self.embedding_manager.embed_texts(texts)

        # 4. Add to vector store
        self.vector_store.add_chunks(chunks, embeddings)

        # 5. Rebuild BM25 with all chunks
        all_chunks = self.vector_store.get_all_chunks()
        self.bm25_index.build_index(all_chunks)

        # 6. Record metadata
        record = DocumentRecord(
            doc_id=identifier,
            filename=file_path.name,
            file_type=file_path.suffix.lower().lstrip("."),
            file_size=file_path.stat().st_size,
            num_pages=len(pages),
            num_chunks=len(chunks),
            uploaded_at=datetime.now().isoformat(),
            status="indexed"
        )
        self.documents[identifier] = record

        # 7. Persist
        self.save_all()
        return record

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False

        record = self.documents.pop(doc_id)

        # Filter out chunks belonging to this document
        remaining_chunks = [c for c in self.vector_store.get_all_chunks() if c.doc_id != doc_id]

        # Rebuild FAISS index
        self.vector_store.clear()
        if remaining_chunks:
            texts = [c.text for c in remaining_chunks]
            embeddings = self.embedding_manager.embed_texts(texts)
            self.vector_store.add_chunks(remaining_chunks, embeddings)

        # Rebuild BM25 index
        self.bm25_index.build_index(remaining_chunks)

        # Remove raw file if present in data/raw
        raw_file = settings.RAW_DIR / record.filename
        if raw_file.exists():
            try:
                raw_file.unlink()
            except Exception as e:
                print(f"Error deleting raw file {raw_file}: {e}")

        self.save_all()
        return True

    def get_documents(self) -> List[DocumentRecord]:
        return list(self.documents.values())

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        return self.documents.get(doc_id)

    def get_chunks_for_document(self, doc_id: str) -> List[DocumentChunk]:
        return [c for c in self.vector_store.get_all_chunks() if c.doc_id != doc_id]

    def total_chunks(self) -> int:
        return self.vector_store.count()
