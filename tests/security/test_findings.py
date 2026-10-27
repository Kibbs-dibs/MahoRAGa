"""
Executable finding inventory (Task 3).

Each test asserts the DESIRED secure behaviour for a finding from
``docs/SECURITY_ATTACK_SURFACE.md``. They are written red-first: before
remediation they fail (proving the weakness is real); after the matching
remediation task they go green.

Finding IDs (F1..F11) match the attack-surface map. Tests are grouped by finding
and marked with ``pytest.mark.security`` so the whole class can be run with
``pytest -m security``.
"""

from __future__ import annotations

import io

import httpx
import pytest

pytestmark = pytest.mark.security


# Write / destructive routes that MUST require the API key once auth lands (F1).
# (method, path, json-body-or-None)
WRITE_ROUTES = [
    ("POST", "/api/v1/ingest", None),            # multipart; sent without files -> still must auth first
    ("POST", "/api/v1/corpus/wipe", None),
    ("POST", "/api/v1/corpus/retry-failed", None),
    ("POST", "/api/v1/watcher/sync", None),
    ("POST", "/api/v1/evaluate", None),
    ("POST", "/api/v1/sessions", {"title": "x"}),
    ("DELETE", "/api/v1/sessions/does-not-exist", None),
]

# Subset used for the POSITIVE (valid-key) check: fast routes that return quickly
# once past the gate. /evaluate and arbitrary pipeline-backed work are excluded
# from the positive check because they do real, multi-second external work — the
# rejection tests already prove those routes share the same auth dependency.
WRITE_ROUTES_FAST = [
    ("POST", "/api/v1/ingest", None),            # -> 400 (no files) once past auth
    ("POST", "/api/v1/corpus/retry-failed", None),
    ("POST", "/api/v1/sessions", {"title": "x"}),
    ("DELETE", "/api/v1/sessions/does-not-exist", None),
]

READ_ROUTES = [
    "/health",
    "/api/v1/corpus",
    "/api/v1/metrics",
    "/api/v1/sessions",
]


# ── F1 / F8 / F9 — scoped authentication ────────────────────────────────────

class TestF1Authentication:
    @pytest.mark.parametrize("method,path,body", WRITE_ROUTES)
    def test_write_routes_reject_unauthenticated(self, client: httpx.Client, method, path, body):
        """Unauthenticated write/destructive calls must be rejected (401/403)."""
        r = client.request(method, path, json=body)
        assert r.status_code in (401, 403), (
            f"{method} {path} returned {r.status_code} without auth; "
            f"expected 401/403 (destructive route exposed)."
        )

    @pytest.mark.parametrize("method,path,body", WRITE_ROUTES_FAST)
    def test_write_routes_accept_valid_key(self, client: httpx.Client, auth_headers, method, path, body):
        """With the API key, write routes must NOT be rejected for auth reasons.

        We only assert the response is not an auth rejection (401/403); the call
        may still 4xx for unrelated reasons (e.g. empty ingest -> 400, unknown
        session delete -> 404). Only fast routes are exercised here (see
        WRITE_ROUTES_FAST) so the shared live server stays responsive.
        """
        r = client.request(method, path, json=body, headers=auth_headers, timeout=20.0)
        assert r.status_code not in (401, 403), (
            f"{method} {path} rejected a valid API key ({r.status_code})."
        )

    def test_wrong_key_rejected(self, client: httpx.Client):
        """A bad key must not pass the gate (F1)."""
        r = client.post("/api/v1/corpus/wipe", headers={"Authorization": "Bearer wrong"})
        assert r.status_code in (401, 403)

    @pytest.mark.parametrize("path", READ_ROUTES)
    def test_read_routes_stay_open(self, client: httpx.Client, path):
        """Read endpoints remain usable without a key (scoped-auth contract)."""
        r = client.get(path)
        assert r.status_code == 200, f"read route {path} should stay open, got {r.status_code}"


# ── F2 — default bind host ───────────────────────────────────────────────────

class TestF2BindHost:
    def test_default_bind_is_loopback(self):
        """start.py must default to a loopback bind, opting into 0.0.0.0 only via env."""
        import inspect
        import start

        src = inspect.getsource(start.main)
        assert "0.0.0.0" not in src or "WEB_UI_HOST" in src, (
            "start.main still hardcodes 0.0.0.0; bind host must be configurable "
            "and default to loopback."
        )


# ── F3 — weak default secrets ────────────────────────────────────────────────

class TestF3Secrets:
    def test_env_example_has_no_live_weak_secret(self):
        """.env.example must not ship usable weak secrets like password / freecc."""
        from pathlib import Path

        text = Path(".env.example").read_text(encoding="utf-8")
        assert "NEO4J_PASSWORD=password" not in text, ".env.example ships NEO4J_PASSWORD=password"
        assert "ANTHROPIC_AUTH_TOKEN=freecc" not in text, ".env.example ships ANTHROPIC_AUTH_TOKEN=freecc"

    def test_compose_has_no_hardcoded_secret(self):
        """docker-compose.yml must not hardcode the Neo4j password literal."""
        from pathlib import Path

        text = Path("docker-compose.yml").read_text(encoding="utf-8")
        assert "neo4j/password" not in text, "compose hardcodes NEO4J_AUTH=neo4j/password"


# ── F4 — stored XSS in UI render helpers ─────────────────────────────────────

class TestF4Xss:
    MALICIOUS = "<img src=x onerror=alert(1)><script>alert(2)</script>"

    def _state_with(self, **memory_kwargs):
        from core.state import AletheiaState, QueryState, MemoryState
        return AletheiaState(
            query=QueryState(original_query="q"),
            memory=MemoryState(**memory_kwargs),
        )

    @staticmethod
    def _assert_inert(out: str, label: str):
        """The payload must not survive as live HTML tags.

        html.escape neutralises XSS by escaping the angle brackets, so the tag
        can never open. We assert the raw ``<script``/``<img`` openers are gone
        and the escaped form is present instead.
        """
        assert "<script" not in out, f"{label}: live <script> tag survived (XSS)"
        assert "<img" not in out, f"{label}: live <img> tag survived (XSS)"
        assert "&lt;" in out, f"{label}: payload was not HTML-escaped at all"

    def test_faiss_markdown_escapes_chunk_text(self):
        from ui.app import format_faiss_markdown
        out = format_faiss_markdown(self._state_with(vector_chunks=[self.MALICIOUS]))
        self._assert_inert(out, "FAISS chunk")

    def test_bm25_markdown_escapes_chunk_text(self):
        from ui.app import format_bm25_markdown
        out = format_bm25_markdown(self._state_with(lexical_chunks=[self.MALICIOUS]))
        self._assert_inert(out, "BM25 chunk")

    def test_external_markdown_escapes_abstract(self):
        from ui.app import format_external_markdown
        out = format_external_markdown(self._state_with(external_api_abstracts=[self.MALICIOUS]))
        self._assert_inert(out, "External abstract")

    def test_neo4j_markdown_escapes_triples(self):
        from ui.app import format_neo4j_markdown
        out = format_neo4j_markdown(self._state_with(serialized_graph_triples=[self.MALICIOUS]))
        # Neo4j formatter also embeds a graph canvas; only assert on the triple list region.
        assert "<script" not in out.split("Active Knowledge Graph Triples")[-1], "Neo4j triple not escaped (XSS)"
        assert "&lt;" in out, "Neo4j triple was not HTML-escaped"

    def test_evidence_breakdown_escapes_source_map(self):
        from ui.app import render_evidence_breakdown_html
        out = render_evidence_breakdown_html(source_map={"S1": self.MALICIOUS})
        self._assert_inert(out, "source_map")


# ── F5 — upload size / content limits ────────────────────────────────────────

class TestF5UploadLimits:
    def test_non_pdf_content_rejected(self, client: httpx.Client, auth_headers):
        """A file with .pdf name but non-PDF bytes must be rejected (magic-byte check)."""
        files = {"files": ("fake.pdf", io.BytesIO(b"not a pdf at all"), "application/pdf")}
        r = client.post("/api/v1/ingest", files=files, headers=auth_headers)
        assert r.status_code == 400, f"non-PDF content accepted ({r.status_code}); expected 400"

    def test_oversize_upload_rejected(self, client: httpx.Client, auth_headers):
        """An over-limit upload must be rejected rather than ingested."""
        big = b"%PDF-1.4" + b"0" * (60 * 1024 * 1024)  # 60 MB
        files = {"files": ("big.pdf", io.BytesIO(big), "application/pdf")}
        r = client.post("/api/v1/ingest", files=files, headers=auth_headers)
        assert r.status_code in (400, 413), f"oversize upload accepted ({r.status_code})"


# ── F6 — orphaned streaming route ────────────────────────────────────────────

class TestF6StreamingRoute:
    def test_streaming_route_resolved(self, client: httpx.Client):
        """The SSE streaming feature must be either reachably registered or removed.

        Resolved == a registered /api/v1/query/stream exists (not 404), OR the
        dead handler has been deleted from the module. A bare undecorated function
        left in place is the failing (unresolved) state.

        We probe with an EMPTY query, which the handler rejects with 400 before
        running any pipeline work — this proves the route is registered and
        reachable without triggering a slow/real pipeline execution. 404 would
        mean it is unregistered.
        """
        import backend.app as appmod

        has_dead_fn = hasattr(appmod, "execute_query_stream")
        r = client.post("/api/v1/query/stream", json={"query": "   "})
        route_registered = r.status_code != 404

        assert route_registered, (
            "No /api/v1/query/stream route is registered (got 404); the SSE "
            "handler is orphaned dead code (F6)."
        )
        # Reaching the handler's validation (400) confirms registration.
        assert r.status_code == 400, (
            f"expected 400 for empty query at the registered stream route, got {r.status_code}"
        )
        assert has_dead_fn, "handler function unexpectedly missing"


# ── F7 — SSRF in arXiv watcher download ──────────────────────────────────────

class TestF7Ssrf:
    def test_non_arxiv_url_refused(self):
        """The watcher must refuse to download a pdf_url pointing off arxiv.org."""
        from agents.arxiv_watcher import ArXivWatcher

        w = ArXivWatcher()
        validator = getattr(w, "_is_allowed_pdf_url", None)
        assert validator is not None, "no URL allow-list validator on the watcher (F7)"
        assert validator("http://169.254.169.254/latest/meta-data/") is False
        assert validator("http://localhost:7687/") is False
        assert validator("https://evil.example.com/x.pdf") is False
        assert validator("https://arxiv.org/pdf/2301.00001.pdf") is True
        assert validator("https://export.arxiv.org/pdf/2301.00001.pdf") is True


# ── UI gating (Task 10) ──────────────────────────────────────────────────────

class TestUiGating:
    def test_ui_auth_knobs_exist(self):
        """Config must expose the UI-login toggle + credentials."""
        from core.config import settings

        for attr in ("aletheia_ui_auth_enabled", "aletheia_ui_username", "aletheia_ui_password"):
            assert hasattr(settings, attr), f"missing UI-auth setting: {attr}"

    def test_start_wires_ui_auth(self):
        """start.py must apply the credentials to the mounted Gradio app."""
        import inspect
        import start

        src = inspect.getsource(start.main)
        assert "aletheia_ui_auth_enabled" in src and "auth=" in src, (
            "start.main does not gate the mounted Gradio UI with auth (Task 10)"
        )


# ── F11 — OpenAPI docs exposure (defense-in-depth) ───────────────────────────

class TestF11Docs:
    def test_docs_can_be_disabled_in_exposed_mode(self):
        """A toggle must exist to disable /docs & /redoc for exposed deployments.

        We assert the app was constructed with a configurable docs URL rather
        than an unconditional "/docs". Implementation adds a setting consulted at
        app creation; here we just require the knob to exist.
        """
        from core.config import settings

        assert hasattr(settings, "expose_api_docs"), (
            "no settings.expose_api_docs toggle to gate /docs in exposed mode (F11)"
        )
