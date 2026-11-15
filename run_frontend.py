import sys
import subprocess
from pathlib import Path
from config import settings

if __name__ == "__main__":
    frontend_script = Path(__file__).resolve().parent / "frontend" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(frontend_script),
        "--server.port",
        str(settings.FRONTEND_PORT),
        "--server.headless",
        "true"
    ]
    print(f"Starting MediRAG Streamlit Frontend on http://localhost:{settings.FRONTEND_PORT}")
    subprocess.run(cmd)
