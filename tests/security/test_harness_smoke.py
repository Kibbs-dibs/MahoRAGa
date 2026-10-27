"""
Smoke test for the live-server security harness (Task 2).

Proves the background uvicorn server boots the unified FastAPI+Gradio app and
answers over the real HTTP stack before any security assertions run.
"""

from __future__ import annotations

import httpx


def test_live_server_boots_and_answers_health(client: httpx.Client):
    """The live server responds 200 on /health over real HTTP."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"


def test_liveness_probe(client: httpx.Client):
    """The liveness probe used by the harness readiness-wait works."""
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json() == {"status": "alive"}


def test_gradio_ui_mounted_at_root(client: httpx.Client):
    """The Gradio UI is served at / (second front door onto the services)."""
    r = client.get("/")
    assert r.status_code == 200
    # Gradio returns an HTML shell.
    assert "text/html" in r.headers.get("content-type", "")
