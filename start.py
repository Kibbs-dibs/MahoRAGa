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
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Bind host (default: WEB_UI_HOST env or 127.0.0.1 loopback). Pass 0.0.0.0 to expose.",
    )
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
    # Bind host is configurable and defaults to loopback so a fresh run is not
    # network-exposed by accident. Set WEB_UI_HOST=0.0.0.0 (or pass --host) to
    # expose on all interfaces — intentionally, not silently.
    from core.config import settings

    bind_host = args.host or getattr(settings, "web_ui_host", "127.0.0.1") or "127.0.0.1"
    display_host = "127.0.0.1" if bind_host in ("0.0.0.0", "::") else bind_host
    url = f"http://{display_host}:{args.port}"
    logger.info("[4/4] Launching Project Aletheia Unified Server on %s (bind=%s)", url, bind_host)
    print(f"\n>> Web Console:  {url}")
    print(f">> OpenAPI Docs: {url}/docs")
    print(f">> Health Check: {url}/health\n")

    # ── Exposure / weak-secret safety warnings ─────────────────────────
    _exposed = bind_host not in ("127.0.0.1", "localhost", "::1")
    if _exposed:
        logger.warning(
            "Server is binding to a NON-loopback interface (%s): it will be "
            "reachable from the network.",
            bind_host,
        )
        if not settings.aletheia_api_key:
            logger.warning(
                "ALETHEIA_API_KEY is not set: write/destructive endpoints will "
                "FAIL CLOSED (HTTP 503). Set it to enable ingest/wipe/watcher."
            )
        weak = []
        if (settings.neo4j_password or "").lower() in ("", "password"):
            weak.append("NEO4J_PASSWORD")
        if (getattr(settings, "anthropic_auth_token", None) or "") == "freecc":
            weak.append("ANTHROPIC_AUTH_TOKEN")
        if weak:
            logger.warning(
                "Weak/default secret(s) detected while network-exposed: %s. "
                "Rotate these before exposing the service.",
                ", ".join(weak),
            )

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

    # Optionally gate the mounted UI with basic auth. The UI calls the service
    # layer directly (not the auth-gated REST routes), so when exposed it must
    # be protected too (recommended for any non-loopback bind).
    ui_auth = None
    if getattr(settings, "aletheia_ui_auth_enabled", False):
        if settings.aletheia_ui_password:
            ui_auth = (settings.aletheia_ui_username, settings.aletheia_ui_password)
            logger.info("Gradio UI authentication ENABLED (user: %s).", settings.aletheia_ui_username)
        else:
            logger.warning(
                "ALETHEIA_UI_AUTH_ENABLED is true but ALETHEIA_UI_PASSWORD is "
                "empty — UI login NOT enforced. Set a password to protect the UI."
            )

    if ui_auth:
        server_app = gr.mount_gradio_app(fastapi_app, gradio_ui, path="/", auth=ui_auth)
    else:
        server_app = gr.mount_gradio_app(fastapi_app, gradio_ui, path="/")

    uvicorn.run(server_app, host=bind_host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
