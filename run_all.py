import sys
import time
import subprocess
from pathlib import Path
from config import settings

def main():
    root = Path(__file__).resolve().parent
    py_exec = sys.executable

    print("=" * 60)
    print("Launching MediRAG Autonomous Healthcare System")
    print("=" * 60)

    # 1. Start Backend
    print(f"[1/2] Starting Backend on http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}...")
    backend_proc = subprocess.Popen([
        py_exec,
        "-m",
        "uvicorn",
        "backend.app:app",
        "--host",
        settings.BACKEND_HOST,
        "--port",
        str(settings.BACKEND_PORT)
    ])

    time.sleep(2)

    # 2. Start Frontend
    print(f"[2/2] Starting Frontend on http://localhost:{settings.FRONTEND_PORT}...")
    frontend_script = root / "frontend" / "app.py"
    frontend_proc = subprocess.Popen([
        py_exec,
        "-m",
        "streamlit",
        "run",
        str(frontend_script),
        "--server.port",
        str(settings.FRONTEND_PORT),
        "--server.headless",
        "true"
    ])

    print("\nMediRAG is LIVE and running!")
    print(f"👉 Frontend UI:  http://localhost:{settings.FRONTEND_PORT}")
    print(f"👉 Backend API:  http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}")
    print(f"👉 API Docs:     http://{settings.BACKEND_HOST}:{settings.BACKEND_PORT}/docs")
    print("\nPress Ctrl+C to terminate both servers.")

    try:
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down MediRAG servers...")
        frontend_proc.terminate()
        backend_proc.terminate()
        frontend_proc.wait()
        backend_proc.wait()
        print("Servers stopped cleanly.")

if __name__ == "__main__":
    main()
