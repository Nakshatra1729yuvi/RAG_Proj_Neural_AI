import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from config import settings
from backend.pipeline import MediRAGPipeline
from backend.storage.document_store import DocumentRecord

app = FastAPI(
    title="MediRAG Healthcare Assistant API",
    description="Autonomous Healthcare RAG Backend with Hybrid Retrieval, Cross-Encoder Reranking, and Grounded Citations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = MediRAGPipeline.get_instance()

class QueryRequest(BaseModel):
    query: str
    search_mode: str = Field(default="hybrid_rerank", description="vector, bm25, hybrid, hybrid_rerank")
    top_k: int = Field(default=settings.TOP_K_FINAL, ge=1, le=15)
    conversation_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)

class QueryResponse(BaseModel):
    answer: str
    raw_answer: str
    is_emergency: bool
    is_supported: bool
    retrieval_mode: str
    citations: List[Dict[str, Any]]
    citation_report: Dict[str, Any]
    evidence_chunks: List[Dict[str, Any]]
    follow_up_questions: List[str]

@app.get("/api/health")
def health_check():
    stats = pipeline.get_stats()
    return {"status": "healthy", "stats": stats}

@app.post("/api/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = pipeline.query(
            question=req.query.strip(),
            search_mode=req.search_mode,
            top_k=req.top_k,
            history=req.conversation_history
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents", response_model=List[DocumentRecord])
def list_documents():
    return pipeline.get_documents()

@app.post("/api/documents/upload", response_model=DocumentRecord)
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed types: .pdf, .docx, .txt"
        )

    # Save to data/raw
    save_path = settings.RAW_DIR / file.filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest into pipeline
        record = pipeline.ingest_file(save_path)
        return record
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    success = pipeline.delete_file(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {"status": "success", "message": f"Document '{doc_id}' deleted successfully."}

@app.get("/api/stats")
def get_stats():
    return pipeline.get_stats()
