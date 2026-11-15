import uvicorn
from config import settings

if __name__ == "__main__":
    print(f"Starting MediRAG FastAPI Backend on http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}")
    uvicorn.run(
        "backend.app:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=False,
        log_level="info"
    )
