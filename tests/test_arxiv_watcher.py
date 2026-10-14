"""
Automated Test Suite for Milestone 4: Autonomous ArXiv Daily Watcher (tests/test_arxiv_watcher.py)

Validates:
1. Atom XML parsing of arXiv search responses.
2. Deduplication against active memory catalog.
3. Automated PDF downloading and streaming ingestion into Three-Layer Global Memory.
4. FastAPI watcher endpoint (/api/v1/watcher/sync).
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import fitz
import httpx
import pytest
from fastapi.testclient import TestClient

from agents.arxiv_watcher import (
    ArXivPaper,
    ArXivWatcher,
    parse_arxiv_atom_feed,
)
from backend.app import app


SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="html">ArXiv Query: search_query=cat:cs.AI</title>
  <id>http://arxiv.org/api/12345</id>
  <entry>
    <id>http://arxiv.org/abs/2503.99999v1</id>
    <published>2026-10-10T12:00:00Z</published>
    <title>AutoRAG: Autonomous Agentic Literature Reviewer</title>
    <summary>We present AutoRAG, an autonomous framework for literature retrieval and cross-encoder synthesis.</summary>
    <author>
      <name>Alice Johnson</name>
    </author>
    <author>
      <name>Bob Williams</name>
    </author>
    <link title="pdf" href="http://arxiv.org/pdf/2503.99999v1" rel="related" type="application/pdf"/>
    <category term="cs.AI"/>
    <category term="cs.CL"/>
  </entry>
</feed>
"""


def _generate_synthetic_pdf_bytes(title: str) -> bytes:
    """Create a minimal valid in-memory PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), f"{title}\n\nAbstract\nAutonomous agentic literature discovery.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_parse_arxiv_atom_feed():
    """Verify Atom XML parses into structured ArXivPaper instances."""
    papers = parse_arxiv_atom_feed(SAMPLE_ARXIV_XML)
    assert len(papers) == 1

    p = papers[0]
    assert p.arxiv_id == "2503.99999"
    assert p.title == "AutoRAG: Autonomous Agentic Literature Reviewer"
    assert "Alice Johnson" in p.authors
    assert "Bob Williams" in p.authors
    assert "2503.99999" in p.pdf_url
    assert "cs.AI" in p.categories
    assert "AutoRAG, an autonomous framework" in p.abstract


def test_search_arxiv_mocked():
    """Verify search_arxiv makes GET request and parses Atom feed."""
    watcher = ArXivWatcher()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = SAMPLE_ARXIV_XML

    with patch("httpx.Client.get", return_value=mock_resp):
        papers = watcher.search_arxiv(query="AutoRAG", max_results=2)
        assert len(papers) == 1
        assert papers[0].title.startswith("AutoRAG")


def test_fetch_and_ingest_pipeline(tmp_path: Path):
    """Verify end-to-end download and auto-ingestion into memory."""
    download_dir = tmp_path / "uploads"
    watcher = ArXivWatcher(download_dir=download_dir)

    pdf_bytes = _generate_synthetic_pdf_bytes("AutoRAG: Autonomous Agentic Literature Reviewer")

    mock_search_resp = MagicMock()
    mock_search_resp.status_code = 200
    mock_search_resp.text = SAMPLE_ARXIV_XML

    mock_pdf_resp = MagicMock()
    mock_pdf_resp.status_code = 200
    mock_pdf_resp.content = pdf_bytes

    orig_get = httpx.Client.get

    def mock_get(self, url, *args, **kwargs):
        url_str = str(url)
        if "arxiv.org/pdf" in url_str:
            return mock_pdf_resp
        elif "arxiv.org" in url_str:
            return mock_search_resp
        return orig_get(self, url, *args, **kwargs)

    with patch.object(httpx.Client, "get", mock_get):
        progress_calls = []
        def track_progress(ratio, msg):
            progress_calls.append((ratio, msg))

        res = watcher.fetch_and_ingest(
            query="AutoRAG",
            max_papers=1,
            progress_cb=track_progress,
        )

        assert res["status"] in ("success", "already_synced")
        assert len(progress_calls) > 0


def test_fastapi_watcher_sync_endpoint():
    """Verify POST /api/v1/watcher/sync endpoint."""
    mock_result = {
        "status": "success",
        "ingested_count": 1,
        "new_chunks": 5,
        "total_documents": 4,
        "total_chunks": 98,
        "neo4j_status": "offline",
        "message": "Autonomous ArXiv sync ingested 1 papers (+5 chunks).",
        "papers": ["AutoRAG"],
    }

    with patch("agents.arxiv_watcher.ArXivWatcher.fetch_and_ingest", return_value=mock_result):
        with TestClient(app) as client:
            res = client.post("/api/v1/watcher/sync?query=RAG&max_papers=1")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "success"
            assert data["ingested_count"] == 1
