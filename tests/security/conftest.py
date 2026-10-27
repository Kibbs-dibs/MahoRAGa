"""
Live-server security test harness (tests/security/conftest.py).

Boots the *unified* application (FastAPI REST + the Gradio UI mounted at ``/``,
exactly as ``start.py`` assembles it) inside a background uvicorn thread on an
ephemeral loopback port, waits for the liveness probe, and hands tests a base
URL plus a ready httpx client.

Why a live server instead of FastAPI's ``TestClient``:
  * exercises the real ASGI stack, CORS middleware, and the Gradio mount at ``/``
  * reflects how a network-exposed deployment actually answers requests
  * lets us assert streaming / SSE behaviour and raw header handling

The harness is deliberately self-contained so the existing in-process
``TestClient`` suites under ``tests/`` keep working unchanged.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Iterator

import httpx
import pytest
import uvicorn

# ── Shared secret used by the scoped-auth tests (Task 4) ────────────────────
# The key is injected into the environment (and the live settings object) by the
# session-scoped ``_security_env`` fixture below — NOT at import time — so it
# does not leak into unrelated test files that share the pytest process.
SECURITY_TEST_API_KEY = "test-secret-key-do-not-use-in-prod"


def _free_port() -> int:
    """Grab an ephemeral TCP port the OS is willing to hand out right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_unified_app():
    """Assemble the same FastAPI+Gradio app object that ``start.py`` serves."""
    import gradio as gr
    from backend.app import app as fastapi_app
    from ui.app import create_ui

    gradio_ui = create_ui()
    return gr.mount_gradio_app(fastapi_app, gradio_ui, path="/")


class _BackgroundServer:
    """Run uvicorn in a daemon thread and expose a clean start/stop lifecycle."""

    def __init__(self, app, host: str, port: int) -> None:
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wait_until_ready(self, base_url: str, timeout: float = 60.0) -> None:
        """Poll the liveness probe until the ASGI app is accepting requests."""
        deadline = time.time() + timeout
        last_err: Exception | None = None
        while time.time() < deadline:
            if getattr(self.server, "should_exit", False):
                raise RuntimeError("Server exited before becoming ready")
            try:
                r = httpx.get(f"{base_url}/health/live", timeout=2.0)
                if r.status_code == 200:
                    return
            except Exception as exc:  # connection refused while booting
                last_err = exc
            time.sleep(0.25)
        raise RuntimeError(f"Live server did not become ready in {timeout}s: {last_err}")

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=10.0)


@pytest.fixture(scope="session", autouse=True)
def _security_env() -> Iterator[None]:
    """Configure auth env for the security session and apply it to live settings.

    Scoped to this session fixture (and restored afterward) so the API key does
    not leak into other test modules sharing the pytest process (e.g. the
    backend TestClient suites that expect loopback-no-key behaviour).
    """
    prev_key = os.environ.get("ALETHEIA_API_KEY")
    prev_host = os.environ.get("WEB_UI_HOST")
    os.environ["ALETHEIA_API_KEY"] = SECURITY_TEST_API_KEY
    os.environ["WEB_UI_HOST"] = "127.0.0.1"

    # The settings singleton may already be imported; push the values onto it so
    # the in-thread server and the auth dependency observe them immediately.
    from core.config import settings
    settings.aletheia_api_key = SECURITY_TEST_API_KEY
    settings.web_ui_host = "127.0.0.1"
    try:
        yield
    finally:
        for var, prev in (("ALETHEIA_API_KEY", prev_key), ("WEB_UI_HOST", prev_host)):
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev
        settings.aletheia_api_key = prev_key
        settings.web_ui_host = prev_host or "127.0.0.1"


@pytest.fixture(scope="session")
def live_server(_security_env) -> Iterator[str]:
    """Session-scoped live server. Yields the base URL (http://127.0.0.1:<port>)."""
    host = "127.0.0.1"
    port = _free_port()
    base_url = f"http://{host}:{port}"

    app = _build_unified_app()
    server = _BackgroundServer(app, host, port)
    server.start()
    try:
        server.wait_until_ready(base_url)
        yield base_url
    finally:
        server.stop()


@pytest.fixture()
def client(live_server: str) -> Iterator[httpx.Client]:
    """httpx client bound to the live server, no auth header by default."""
    with httpx.Client(base_url=live_server, timeout=30.0) as c:
        yield c


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    """Authorization header carrying the configured test API key (Task 4)."""
    return {"Authorization": f"Bearer {SECURITY_TEST_API_KEY}"}


@pytest.fixture()
def api_key() -> str:
    """The shared test API key, for tests that build their own requests."""
    return SECURITY_TEST_API_KEY
