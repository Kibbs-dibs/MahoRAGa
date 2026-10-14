"""
Project Aletheia — Python Entrypoint & Launcher (start.py)

Usage:
    python start.py [--port 7860] [--ingest] [--no-browser]
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

# ── Auto-delegate to .venv python if invoked with global python ──────────────
_venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    print(f"[*] Detected project virtual environment. Switching to {_venv_python}...")
    sys.exit(subprocess.call([str(_venv_python)] + sys.argv))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("start")


def check_port(host: str, port: int) -> bool:
    """Check if a network daemon is listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def print_banner():
    banner = """
======================================================================
                    PROJECT ALETHEIA
      Autonomous Multi-Agent AI Literature Research Assistant
======================================================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="Project Aletheia Launcher")
    parser.add_argument("--port", type=int, default=7860, help="Web UI port (default: 7860)")
    parser.add_argument("--ingest", action="store_true", help="Force rebuild corpus indexes")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    print_banner()

    # ── 1. Check & Build Corpus Indexes ─────────────────────────────────
    faiss_file = Path("data/faiss_index.bin")
    bm25_file = Path("data/bm25_index.pkl")

    if args.ingest or not faiss_file.exists() or not bm25_file.exists():
        logger.info("[1/4] Corpus indexes missing or --ingest specified. Running ingestion...")
        from ingestion.cli import run_ingestion_pipeline

        summary = run_ingestion_pipeline(
            input_dir="RAGxamples",
            output_dir="data",
            index_neo4j=False,
        )
        logger.info("[1/4] Ingestion complete: %d chunks indexed.", summary.get("total_chunks", 0))
    else:
        logger.info("[1/4] Memory indexes verified (data/faiss_index.bin, data/bm25_index.pkl).")

    # ── 2. Check Local Daemons ──────────────────────────────────────────
    neo4j_online = check_port("localhost", 7687)
    if neo4j_online:
        logger.info("[2/4] Neo4j Daemon active on bolt://localhost:7687.")
    else:
        logger.info("[2/4] Neo4j is offline on localhost:7687 (Graceful local fallback active).")

    ollama_online = check_port("localhost", 11434)
    if ollama_online:
        logger.info("[3/4] Ollama Daemon active on http://localhost:11434.")
    else:
        logger.info("[3/4] Ollama is offline on localhost:11434 (Deterministic fallback active).")

    # ── 3. Launch Unified FastAPI + Gradio Web Console ─────────────────
    url = f"http://127.0.0.1:{args.port}"
    logger.info("[4/4] Launching Project Aletheia Unified Server on %s", url)
    print(f"\n>> Web Console:  {url}")
    print(f">> OpenAPI Docs: {url}/docs")
    print(f">> Health Check: {url}/health\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import gradio as gr
    import uvicorn
    from backend.app import app as fastapi_app
    from ui.app import create_ui

    gradio_ui = create_ui()
    server_app = gr.mount_gradio_app(fastapi_app, gradio_ui, path="/")

    uvicorn.run(server_app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
