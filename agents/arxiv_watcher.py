"""
Autonomous ArXiv Daily Watcher Agent for Project Aletheia (agents/arxiv_watcher.py)

Autonomous literature harvesting agent that:
1. Queries the official arXiv Atom REST API for AI literature (cs.AI, cs.CL, cs.LG).
2. Parses metadata, abstract summaries, authors, and canonical PDF URLs.
3. Automatically deduplicates against active catalog memory (data/corpus_catalog.json).
4. Downloads candidate PDFs into data/uploads/.
5. Automatically streams newly discovered papers through LiveIngestionService into Three-Layer Global Memory.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from ingestion.service import ingestion_service

logger = logging.getLogger("agents.arxiv_watcher")

ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArXivPaper(BaseModel):
    """Metadata representation of an academic preprint fetched from arXiv."""
    arxiv_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str
    published: str
    pdf_url: str
    categories: List[str] = Field(default_factory=list)


def parse_arxiv_atom_feed(xml_text: str) -> List[ArXivPaper]:
    """Parse arXiv Atom XML response into structured ArXivPaper instances."""
    try:
        root = ET.fromstring(xml_text)
    except Exception as exc:
        logger.error("Failed to parse arXiv Atom XML: %s", exc)
        return []

    papers: List[ArXivPaper] = []

    for entry in root.findall(f"{ATOM_NS}entry"):
        id_elem = entry.find(f"{ATOM_NS}id")
        raw_id = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
        arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
        # Strip only a trailing version suffix like "v2" (not any 'v' in the id).
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        title_elem = entry.find(f"{ATOM_NS}title")
        raw_title = title_elem.text if title_elem is not None and title_elem.text else "Untitled"
        title = re.sub(r"\s+", " ", raw_title).strip()

        summary_elem = entry.find(f"{ATOM_NS}summary")
        raw_summary = summary_elem.text if summary_elem is not None and summary_elem.text else ""
        abstract = re.sub(r"\s+", " ", raw_summary).strip()

        pub_elem = entry.find(f"{ATOM_NS}published")
        published = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else ""

        authors = []
        for author in entry.findall(f"{ATOM_NS}author"):
            name_elem = author.find(f"{ATOM_NS}name")
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        # PDF URL extraction
        pdf_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break

        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        elif pdf_url and not pdf_url.endswith(".pdf"):
            pdf_url = f"{pdf_url}.pdf"

        # arXiv feed hrefs are sometimes http://; upgrade to https so the
        # download SSRF allow-list (https-only) accepts legitimate arXiv PDFs.
        if pdf_url.startswith("http://") and (
            "//arxiv.org" in pdf_url or "//export.arxiv.org" in pdf_url
        ):
            pdf_url = "https://" + pdf_url[len("http://"):]

        # Categories
        categories = []
        for cat in entry.findall(f"{ATOM_NS}category"):
            term = cat.attrib.get("term")
            if term:
                categories.append(term)

        if arxiv_id and title:
            papers.append(
                ArXivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published=published,
                    pdf_url=pdf_url,
                    categories=categories,
                )
            )

    return papers


class ArXivWatcher:
    """Agent that watches arXiv categories and autonomously synchronizes literature."""

    def __init__(
        self,
        default_categories: Optional[List[str]] = None,
        download_dir: str | Path = "data/uploads",
    ) -> None:
        self.categories = default_categories or ["cs.AI", "cs.CL", "cs.LG"]
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.api_base_url = "http://export.arxiv.org/api/query"

    # Hosts the watcher is permitted to download PDFs from. The pdf_url comes
    # from fetched Atom XML (attacker-influenceable), so restrict it to arXiv
    # rather than letting it point at internal/metadata endpoints (SSRF — F7).
    _ALLOWED_PDF_HOSTS = frozenset({"arxiv.org", "export.arxiv.org"})
    _MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB download cap

    @classmethod
    def _is_allowed_pdf_url(cls, url: str) -> bool:
        """Return True only for https URLs whose host is an allowed arXiv host."""
        try:
            parsed = urllib.parse.urlparse((url or "").strip())
        except Exception:
            return False
        if parsed.scheme != "https":
            return False
        host = (parsed.hostname or "").lower()
        return host in cls._ALLOWED_PDF_HOSTS

    def search_arxiv(
        self,
        query: str = "Retrieval-Augmented Generation",
        max_results: int = 5,
    ) -> List[ArXivPaper]:
        """
        Query the official arXiv API for papers matching categories and keywords.
        """
        cat_filter = " OR ".join(f"cat:{cat}" for cat in self.categories)
        clean_query = query.strip()
        search_query = f"({cat_filter}) AND all:{clean_query}" if clean_query else cat_filter

        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": max_results,
        }

        url = f"{self.api_base_url}?{urllib.parse.urlencode(params)}"
        logger.info("Querying arXiv API: %s", url)

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(url)
                if resp.status_code != 200:
                    logger.warning("arXiv API returned non-200 status: %d", resp.status_code)
                    return []
                return parse_arxiv_atom_feed(resp.text)
        except Exception as exc:
            logger.error("arXiv API request failed: %s", exc)
            return []

    def fetch_and_ingest(
        self,
        query: str = "Retrieval-Augmented Generation",
        max_papers: int = 3,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Search arXiv, download candidate PDFs, and auto-ingest into memory.
        """
        def update_progress(ratio: float, msg: str):
            if progress_cb:
                try:
                    progress_cb(ratio, msg)
                except Exception:
                    pass

        update_progress(0.1, f"Searching arXiv for '{query}'...")
        papers = self.search_arxiv(query=query, max_results=max_papers * 2)

        if not papers:
            return {
                "status": "no_results",
                "ingested_count": 0,
                "message": f"No arXiv preprints found matching query '{query}'.",
                "papers": [],
            }

        catalog = ingestion_service.get_catalog()
        # Dedup on the doc_id that ingestion will assign. The watcher saves each PDF
        # as "{sanitized_arxiv_id}.pdf", and the ingestion service derives doc_id from
        # the filename stem — so compare on the same sanitized arxiv_id instead of a
        # fragile title-substring test, keeping watcher and ingestion in agreement
        # (Issue 27). Exact (normalised) titles are kept as a secondary guard.
        def _safe_id(arxiv_id: str) -> str:
            return re.sub(r"[^\w\-.]", "_", arxiv_id)

        known_doc_ids = {d.get("doc_id") for d in catalog.get("documents", [])}
        known_titles = {d.get("title", "").strip().lower() for d in catalog.get("documents", [])}

        downloaded_paths: List[Path] = []
        selected_papers: List[ArXivPaper] = []

        # Filter out papers already ingested (by arXiv-id-derived doc_id, then exact title)
        for p in papers:
            if _safe_id(p.arxiv_id) in known_doc_ids:
                logger.info("Skipping paper already in catalog (id match): %s", p.arxiv_id)
                continue
            if p.title.strip().lower() in known_titles:
                logger.info("Skipping paper already in catalog (title match): %s", p.title)
                continue

            selected_papers.append(p)
            if len(selected_papers) >= max_papers:
                break

        if not selected_papers:
            return {
                "status": "already_synced",
                "ingested_count": 0,
                "message": f"All discovered arXiv preprints for '{query}' are already in memory.",
                "papers": [p.title for p in papers[:max_papers]],
            }

        # Download PDF files
        total_p = len(selected_papers)
        for idx, paper in enumerate(selected_papers, 1):
            ratio = 0.2 + (idx / total_p) * 0.35
            update_progress(ratio, f"Downloading [{idx}/{total_p}] {paper.title[:35]}...")

            safe_name = re.sub(r"[^\w\-.]", "_", paper.arxiv_id) + ".pdf"
            dest_file = self.download_dir / safe_name

            # SSRF guard: only download from allowed arXiv hosts over https (F7).
            if not self._is_allowed_pdf_url(paper.pdf_url):
                logger.warning(
                    "Refusing non-arXiv / non-https PDF URL for %s: %s",
                    paper.arxiv_id,
                    paper.pdf_url,
                )
                continue

            try:
                # follow_redirects=False so a crafted redirect cannot bounce the
                # request to an internal host after the allow-list check.
                with httpx.Client(timeout=30.0, follow_redirects=False) as client:
                    pdf_resp = client.get(paper.pdf_url)
                    # Any redirect must also stay on an allowed host. Gate on the
                    # numeric status code (robust vs. httpx internals) rather than
                    # the is_redirect property.
                    if isinstance(pdf_resp.status_code, int) and 300 <= pdf_resp.status_code < 400:
                        loc = pdf_resp.headers.get("location", "")
                        if not self._is_allowed_pdf_url(loc):
                            logger.warning("Refusing off-arxiv redirect for %s: %s", paper.arxiv_id, loc)
                            continue
                        pdf_resp = client.get(loc)
                    content = pdf_resp.content
                    if len(content) > self._MAX_PDF_BYTES:
                        logger.warning(
                            "Refusing oversized PDF for %s (%d bytes > cap)",
                            paper.arxiv_id,
                            len(content),
                        )
                        continue
                    if pdf_resp.status_code == 200 and len(content) > 100 and content.startswith(b"%PDF"):
                        with open(dest_file, "wb") as fh:
                            fh.write(content)
                        downloaded_paths.append(dest_file)
                    else:
                        logger.warning(
                            "Failed to download PDF for %s (Status: %d, Size: %d bytes)",
                            paper.arxiv_id,
                            pdf_resp.status_code,
                            len(content),
                        )
            except Exception as exc:
                logger.error("Download failed for %s: %s", paper.arxiv_id, exc)

        if not downloaded_paths:
            return {
                "status": "download_failed",
                "ingested_count": 0,
                "message": "Failed to download candidate PDFs from arXiv.",
                "papers": [p.title for p in selected_papers],
            }

        # Ingest downloaded PDFs into Three-Layer Global Memory
        update_progress(0.65, f"Ingesting {len(downloaded_paths)} arXiv papers into memory...")

        def ingest_step_tracker(sub_ratio: float, desc: str):
            overall = 0.65 + (sub_ratio * 0.33)
            update_progress(overall, desc)

        ingest_res = ingestion_service.ingest_files(
            downloaded_paths,
            progress_cb=ingest_step_tracker,
        )

        update_progress(1.0, "ArXiv sync complete!")

        return {
            "status": "success",
            "ingested_count": ingest_res.get("ingested_count", 0),
            "new_chunks": ingest_res.get("new_chunks", 0),
            "total_documents": ingest_res.get("total_documents", 0),
            "total_chunks": ingest_res.get("total_chunks", 0),
            "neo4j_status": ingest_res.get("neo4j_status", "skipped"),
            "message": f"Autonomous ArXiv sync ingested {ingest_res.get('ingested_count', 0)} papers (+{ingest_res.get('new_chunks', 0)} chunks).",
            "papers": [p.title for p in selected_papers],
        }


# Singleton instance
arxiv_watcher = ArXivWatcher()
