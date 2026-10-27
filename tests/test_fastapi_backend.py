"""
Automated Test Suite for Milestone 3: FastAPI Backend & Session Persistence (tests/test_fastapi_backend.py)

Validates:
1. Health check endpoint (/health)
2. Session CRUD operations (/api/v1/sessions)
3. End-to-end query execution with session logging (/api/v1/query)
4. Corpus catalog inspection (/api/v1/corpus)
5. Graph topology export (/api/v1/graph)
6. Multipart PDF file ingestion endpoint (/api/v1/ingest)
"""

from __future__ import annotations

import io
from pathlib import Path
import fitz
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.database import db_manager


@pytest.fixture(scope="module")
def client():
    """Create FastAPI TestClient.

    Write/destructive routes are auth-gated (backend/auth.py). When an
    ALETHEIA_API_KEY is configured in the environment (e.g. when the security
    suite runs in the same process), attach it by default so these functional
    tests keep exercising the handlers rather than the auth gate.
    """
    from core.config import settings

    headers = {}
    if settings.aletheia_api_key:
        headers["Authorization"] = f"Bearer {settings.aletheia_api_key}"
    with TestClient(app, headers=headers) as test_client:
        yield test_client


def test_health_check(client: TestClient):
    """Verify /health returns system status and catalog counts."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "faiss_ready" in data
    assert "bm25_ready" in data
    assert "total_documents" in data


def test_session_lifecycle(client: TestClient):
    """Verify session creation, message persistence, listing, and deletion."""
    # 1. Create session
    create_res = client.post("/api/v1/sessions", json={"title": "Test AttentionRAG Investigation"})
    assert create_res.status_code == 200
    session_data = create_res.json()
    sess_id = session_data["id"]
    assert sess_id.startswith("sess_")
    assert session_data["title"] == "Test AttentionRAG Investigation"

    # 2. List sessions
    list_res = client.get("/api/v1/sessions")
    assert list_res.status_code == 200
    sessions = list_res.json()
    assert any(s["id"] == sess_id for s in sessions)

    # 3. Add message manually via db_manager
    db_manager.add_message(
        session_id=sess_id,
        role="user",
        content="How does AttentionRAG work?",
    )
    db_manager.add_message(
        session_id=sess_id,
        role="assistant",
        content="AttentionRAG compresses context.",
        rubric_score=92,
        compression_ratio=5.4,
    )

    # 4. Get session detail
    detail_res = client.get(f"/api/v1/sessions/{sess_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["id"] == sess_id
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"
    assert detail["messages"][1]["rubric_score"] == 92

    # 5. Delete session
    del_res = client.delete(f"/api/v1/sessions/{sess_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Verify not found after deletion
    detail_res_after = client.get(f"/api/v1/sessions/{sess_id}")
    assert detail_res_after.status_code == 404


def test_query_endpoint(client: TestClient):
    """Verify /api/v1/query runs LangGraph and persists conversation messages."""
    payload = {
        "query": "How does AttentionRAG compress context?",
        "evidence_threshold": 80.0,
        "max_compression_ratio": 6.3,
        "enable_external_search": False,
    }

    res = client.post("/api/v1/query", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "session_id" in data
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "evidence_score" in data
    assert "compression_ratio" in data
    assert "critic_verdict" in data
    assert "latency_seconds" in data

    # Verify message was logged to the session in SQLite
    sess_id = data["session_id"]
    sess_res = client.get(f"/api/v1/sessions/{sess_id}")
    assert sess_res.status_code == 200
    messages = sess_res.json()["messages"]
    assert len(messages) >= 2
    assert messages[-1]["role"] == "assistant"


def test_query_empty_reject(client: TestClient):
    """Verify empty queries return 400 Bad Request."""
    res = client.post("/api/v1/query", json={"query": "   "})
    assert res.status_code == 400


def test_corpus_and_graph_endpoints(client: TestClient):
    """Verify corpus catalog and knowledge graph topology endpoints."""
    # Corpus
    corpus_res = client.get("/api/v1/corpus")
    assert corpus_res.status_code == 200
    corpus = corpus_res.json()
    assert "total_documents" in corpus
    assert "documents" in corpus

    # Graph
    graph_res = client.get("/api/v1/graph")
    assert graph_res.status_code == 200
    graph = graph_res.json()
    assert "nodes" in graph
    assert "edges" in graph


def test_ingest_endpoint(client: TestClient, tmp_path: Path):
    """Verify multipart PDF file upload endpoint."""
    # Generate a lightweight test PDF in memory
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "FastAPI Benchmark Paper\n\nAbstract\nDemonstrating multipart PDF upload into Aletheia API.")
    pdf_bytes = doc.tobytes()
    doc.close()

    files = [
        ("files", ("fastapi_test_paper.pdf", io.BytesIO(pdf_bytes), "application/pdf"))
    ]

    res = client.post("/api/v1/ingest", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("success", "partial_success", "skipped")
    assert "total_documents" in data
    assert "total_chunks" in data
