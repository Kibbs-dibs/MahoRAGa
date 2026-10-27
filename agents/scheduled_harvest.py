"""Scheduled autonomous arXiv harvest for MahoRAGa (Feature 6.4).

Runs the ArXivWatcher for a set of tracked topics and ingests new papers into
the Three-Layer Global Memory, turning the corpus into a living daily feed.

Usage:
    python -m agents.scheduled_harvest                 # default topics
    python -m agents.scheduled_harvest "RAG" "agents"  # explicit topics

Intended to be invoked by a scheduler (cron / Kiro Crew cron). It is safe to run
repeatedly: ingestion dedups by arXiv-id-derived doc_id and MD5, so already-seen
papers are skipped.
"""
from __future__ import annotations

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agents.scheduled_harvest")

DEFAULT_TOPICS = [
    "retrieval augmented generation",
    "large language model agents",
    "knowledge graph reasoning",
]


def run(topics: list[str] | None = None, max_papers_per_topic: int = 3) -> dict:
    from agents.arxiv_watcher import arxiv_watcher

    topics = topics or DEFAULT_TOPICS
    summary = {"topics": {}, "total_ingested": 0, "total_new_chunks": 0}
    for topic in topics:
        try:
            res = arxiv_watcher.fetch_and_ingest(query=topic, max_papers=max_papers_per_topic)
            ingested = res.get("ingested_count", 0)
            chunks = res.get("new_chunks", 0)
            summary["topics"][topic] = {"status": res.get("status"), "ingested": ingested, "new_chunks": chunks}
            summary["total_ingested"] += ingested
            summary["total_new_chunks"] += chunks
            logger.info("Harvested '%s': %s (+%d papers, +%d chunks)", topic, res.get("status"), ingested, chunks)
        except Exception as exc:
            logger.warning("Harvest failed for '%s': %s", topic, exc)
            summary["topics"][topic] = {"status": "error", "error": str(exc)}
    logger.info("Scheduled harvest complete: +%d papers, +%d chunks across %d topics",
                summary["total_ingested"], summary["total_new_chunks"], len(topics))
    return summary


if __name__ == "__main__":
    args = sys.argv[1:]
    run(args or None)
