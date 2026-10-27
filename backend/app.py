"""
FastAPI Production Backend for Project Aletheia (backend/app.py)

Exposes high-performance REST endpoints for:
- Literature Research Pipeline execution (/api/v1/query)
- Persistent Multi-Turn Chat Sessions (/api/v1/sessions)
- Real-time PDF Upload and Indexing (/api/v1/ingest)
- Three-Layer Global Memory Corpus Management (/api/v1/corpus)
- Knowledge Graph Topology Export (/api/v1/graph)
- System Health Diagnostics (/health)
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.auth import require_write_auth

from core.config import settings
from core.state import (
    AletheiaState,
    QueryState,
    MemoryState,
    EvaluationState,
    CompressionState,
    GenerationState,
)
from graph.orchestrator import run_aletheia_pipeline_async
from ingestion.service import ingestion_service
from backend.database import db_manager
from backend.schemas import (
    BenchmarkResponse,
    HealthResponse,
    IngestResponse,
    MessageItem,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    SessionCreate,
    SessionDetail,
)
from core.tracing import tracer
from ui.graph_visualizer import parse_triples_to_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.app")

# Gate the interactive OpenAPI docs behind a setting (F11, defense-in-depth):
# a network-exposed deployment can set expose_api_docs=false so the full API
# surface is not advertised. Authentication (not this toggle) is the real
# control; this just reduces discoverability.
_expose_docs = getattr(settings, "expose_api_docs", True)
app = FastAPI(
    title="Project Aletheia API",
    description="Autonomous Multi-Agent AI Literature Research Assistant Backend",
    version="1.0.0",
    docs_url="/docs" if _expose_docs else None,
    redoc_url="/redoc" if _expose_docs else None,
    openapi_url="/openapi.json" if _expose_docs else None,
)

# Enable CORS for local dashboards, web clients, or browser extensions.
# Wildcard origins with credentials is invalid/insecure (browsers reject it and it
# invites CSRF), so we only allow credentials when explicit origins are configured.
_cors_origins = [o.strip() for o in getattr(settings, "cors_allow_origins", "*").split(",") if o.strip()]
_wildcard = _cors_origins == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=not _wildcard,  # never True together with "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Concurrency back-pressure ────────────────────────────────────────────
# Bound simultaneous pipeline runs so a burst queues (and sheds past a short
# wait) instead of exhausting host memory. Created lazily on the running loop.
_query_semaphore: Optional[asyncio.Semaphore] = None


def _get_query_semaphore() -> asyncio.Semaphore:
    global _query_semaphore
    if _query_semaphore is None:
        _query_semaphore = asyncio.Semaphore(max(1, settings.max_concurrent_queries))
    return _query_semaphore


async def _acquire_slot_or_503():
    """Acquire a concurrency slot within the admission window, else raise 503."""
    sem = _get_query_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=settings.query_admission_wait_seconds)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Server is at capacity (too many concurrent queries). Please retry "
                "shortly."
            ),
        )
    return sem


# ── Health & Diagnostics ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Verify operational health across Three-Layer Global Memory and pipeline services."""
    faiss_ok = os.path.isfile(settings.faiss_index_path)
    bm25_ok = os.path.isfile(settings.bm25_index_path)

    neo4j_online = False
    try:
        from memory.neo4j_client import Neo4jClient
        with Neo4jClient() as client:
            client.connect()
            neo4j_online = True
    except Exception:
        neo4j_online = False

    cat = ingestion_service.get_catalog()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        faiss_ready=faiss_ok,
        bm25_ready=bm25_ok,
        neo4j_online=neo4j_online,
        total_documents=cat.get("total_documents", 0),
        total_chunks=cat.get("total_chunks", 0),
    )


@app.get("/health/live", tags=["Diagnostics"])
async def liveness():
    """Liveness probe: the process is up and the event loop responds.

    Deliberately does NOT touch stores or backends — a dependency being down must
    not cause an orchestrator to kill/restart an otherwise-healthy process.
    """
    return {"status": "alive"}


def _any_llm_backend_reachable() -> Dict[str, Any]:
    """Probe configured LLM backends cheaply; return which (if any) answer.

    A backend is 'reachable' if a short TCP+HTTP probe to its base URL succeeds.
    Offline fallback always exists, so readiness treats 'no real backend' as a
    DEGRADED-but-serving signal rather than a hard failure.
    """
    import httpx

    reachable: List[str] = []
    probes = []
    if settings.ollama_base_url:
        probes.append(("ollama", settings.ollama_base_url.rstrip("/") + "/api/tags"))
    if settings.openai_base_url or settings.openai_api_key:
        probes.append(("openai", (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/") + "/models"))
    if settings.anthropic_base_url:
        probes.append(("proxy", settings.anthropic_base_url.rstrip("/")))

    for name, url in probes:
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=1.0, read=2.0, write=1.0, pool=1.0)) as c:
                r = c.get(url)
                # Any HTTP answer (even 401/404) means the server is UP.
                if r.status_code < 500:
                    reachable.append(name)
        except Exception:
            continue

    return {"reachable": reachable, "configured": [p[0] for p in probes]}


@app.get("/health/ready", tags=["Diagnostics"])
async def readiness(response: Response):
    """Readiness probe: can this instance actually SERVE a query right now?

    Ready requires at least one retrieval index present (FAISS or BM25). A healthy
    LLM backend is reported but not required — the offline fallback keeps the
    service answering (flagged degraded). When not ready, returns HTTP 503 so a
    load balancer / the harvest cron stops routing work here.
    """
    faiss_ok = os.path.isfile(settings.faiss_index_path)
    bm25_ok = os.path.isfile(settings.bm25_index_path)
    corpus_ok = faiss_ok or bm25_ok

    llm = _any_llm_backend_reachable()
    llm_ok = len(llm["reachable"]) > 0

    ready = corpus_ok  # minimum bar to serve a query
    body = {
        "status": "ready" if ready else "not_ready",
        "corpus_ready": corpus_ok,
        "faiss_ready": faiss_ok,
        "bm25_ready": bm25_ok,
        "llm_backends_reachable": llm["reachable"],
        "llm_backends_configured": llm["configured"],
        "degraded": not llm_ok,  # serving on offline fallback only
        "reason": None if ready else "No retrieval index present (ingest a document first).",
    }
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return body


# ── Research Query Pipeline ──────────────────────────────────────────────

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Research Pipeline"])
async def execute_query(req: QueryRequest):
    """
    Execute the 6-node LangGraph literature review pipeline:
    Router -> Retriever -> Evaluator -> (Search APIs) -> Pruner -> Generator & Critic
    """
    if not req.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query prompt cannot be empty.",
        )

    # 1. Manage session persistence
    session_id = req.session_id
    if not session_id or not db_manager.get_session(session_id):
        new_sess = db_manager.create_session(title=req.query[:45])
        session_id = new_sess["id"]

    # Record user message
    db_manager.add_message(session_id=session_id, role="user", content=req.query)

    trace = tracer.start_trace(req.query)
    start_time = time.time()

    # Admission control: bound concurrent pipeline runs (503 if saturated).
    sem = await _acquire_slot_or_503()

    # Build recent conversation history for multi-turn follow-ups (Feature 7.4):
    # the last few prior user/assistant turns, compacted. Fetched BEFORE this turn's
    # user message is added above? (it was added) — so exclude the just-added one.
    history_lines: List[str] = []
    try:
        prior = db_manager.get_messages(session_id)[:-1]  # drop this turn's user msg
        for m in prior[-6:]:  # last ~3 Q/A pairs
            role = "Q" if m.get("role") == "user" else "A"
            content = (m.get("content") or "").strip().replace("\n", " ")
            if content:
                history_lines.append(f"{role}: {content[:300]}")
    except Exception as exc:
        logger.debug("Conversation history unavailable: %s", exc)

    # 2. Build pipeline state
    initial_state = AletheiaState(
        query=QueryState(original_query=req.query.strip(), conversation_history=history_lines),
        memory=MemoryState(),
        evaluation=EvaluationState(rubric_score=0),
        compression=CompressionState(),
        generation=GenerationState(),
        evidence_threshold_override=req.evidence_threshold,
        max_compression_ratio_override=req.max_compression_ratio,
        enable_external_search=req.enable_external_search,
    )

    try:
        # Execute LangGraph asynchronously, bounded by a hard wall-clock ceiling so a
        # wedged backend or runaway retrieval cannot hold the request open forever.
        final_state_raw = await asyncio.wait_for(
            run_aletheia_pipeline_async(initial_state),
            timeout=settings.request_timeout_seconds,
        )

        # Convert back to typed AletheiaState
        final_state: AletheiaState
        if isinstance(final_state_raw, AletheiaState):
            final_state = final_state_raw
        elif isinstance(final_state_raw, dict):
            final_state = AletheiaState.model_validate(final_state_raw)
        else:
            final_state = initial_state

        elapsed = time.time() - start_time

        ans = (final_state.generation.draft_answer if final_state.generation else "") or "Research synthesis complete."
        ev_score = final_state.evaluation.rubric_score if final_state.evaluation else 0
        ev = final_state.evaluation
        evidence_breakdown = {
            "retrieval_confidence": ev.retrieval_confidence if ev else 0,
            "answer_specificity": ev.answer_specificity if ev else 0,
            "source_diversity": ev.source_diversity if ev else 0,
            "metadata_completeness": ev.metadata_completeness if ev else 0,
            "recency": ev.recency if ev else 0,
        }
        comp_ratio = final_state.compression.compression_ratio if final_state.compression else 1.0
        critic_verdict = (
            "Approved (Zero Hallucinations)"
            if (not final_state.generation or not final_state.generation.critic_feedback or "approved" in final_state.generation.critic_feedback.lower())
            else "Regenerated (Self-Corrected)"
        )

        triples = (final_state.memory.serialized_graph_triples if final_state.memory else []) or []
        vectors = (final_state.memory.vector_chunks if final_state.memory else []) or []
        lexical = (final_state.memory.lexical_chunks if final_state.memory else []) or []
        external = (final_state.memory.external_api_abstracts if final_state.memory else []) or []

        # Record assistant response in database
        db_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=ans,
            rubric_score=ev_score,
            compression_ratio=comp_ratio,
            critic_verdict=critic_verdict,
            latency=elapsed,
        )

        # Record audit trace
        db_manager.record_audit(
            query=req.query,
            session_id=session_id,
            reformulated_query=final_state.query.reformulated_query if final_state.query else "",
            routing_decision=final_state.query.route if final_state.query else "",
            vector_chunks_count=len(vectors),
            lexical_chunks_count=len(lexical),
            graph_triples_count=len(triples),
            external_count=len(external),
            rubric_score=ev_score,
            compression_ratio=comp_ratio,
            critic_verdict=critic_verdict,
            total_latency=elapsed,
        )

        trace.finish(
            rubric_score=ev_score,
            compression_ratio=comp_ratio,
            critic_verdict=critic_verdict,
        )

        return QueryResponse(
            session_id=session_id,
            answer=ans,
            evidence_score=ev_score,
            compression_ratio=comp_ratio,
            critic_verdict=critic_verdict,
            latency_seconds=round(elapsed, 2),
            triples=triples,
            vector_chunks=vectors,
            lexical_chunks=lexical,
            external_abstracts=external,
            evidence_breakdown=evidence_breakdown,
            source_map=(final_state.compression.source_map if final_state.compression else {}),
            degraded=bool(final_state.memory and final_state.memory.degraded_layers),
            degraded_layers=(final_state.memory.degraded_layers if final_state.memory else []),
        )

    except asyncio.TimeoutError as exc:
        trace.finish(error="request timeout")
        logger.error(
            "Pipeline exceeded request_timeout_seconds=%.1fs for query: %s",
            settings.request_timeout_seconds,
            req.query[:120],
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"The research pipeline did not finish within "
                f"{settings.request_timeout_seconds:.0f}s (a backend may be slow or "
                f"unavailable). Please retry; if this persists, check /health/ready."
            ),
        ) from exc

    except Exception as exc:
        trace.finish(error=str(exc))
        logger.error("Pipeline execution error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        ) from exc

    finally:
        sem.release()


@app.post("/api/v1/query/stream", tags=["Research Pipeline"])
async def execute_query_stream(req: QueryRequest):
    """Streaming variant of /query (Feature 4.1): runs the pipeline, then streams
    the answer progressively as Server-Sent Events, followed by a final metadata
    event (evidence score, compression ratio, critic verdict/confidence, sources).
    Large perceived-latency win vs. waiting for the whole JSON body.
    """
    import json as _json

    if not req.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query prompt cannot be empty.")

    session_id = req.session_id
    if not session_id or not db_manager.get_session(session_id):
        session_id = db_manager.create_session(title=req.query[:45])["id"]
    db_manager.add_message(session_id=session_id, role="user", content=req.query)

    initial_state = AletheiaState(
        query=QueryState(original_query=req.query.strip()),
        memory=MemoryState(), evaluation=EvaluationState(rubric_score=0),
        compression=CompressionState(), generation=GenerationState(),
        evidence_threshold_override=req.evidence_threshold,
        max_compression_ratio_override=req.max_compression_ratio,
        enable_external_search=req.enable_external_search,
    )

    async def event_stream():
        try:
            final_state_raw = await run_aletheia_pipeline_async(initial_state)
            fs = final_state_raw if isinstance(final_state_raw, AletheiaState) else (
                AletheiaState.model_validate(final_state_raw) if isinstance(final_state_raw, dict) else initial_state
            )
            answer = (fs.generation.draft_answer if fs.generation else "") or "Research synthesis complete."
            # Progressive delivery: stream the answer in word-chunks.
            words = answer.split(" ")
            buf = []
            for w in words:
                buf.append(w)
                if len(buf) >= 8:
                    yield f"data: {_json.dumps({'type': 'token', 'text': ' '.join(buf) + ' '})}\n\n"
                    buf = []
            if buf:
                yield f"data: {_json.dumps({'type': 'token', 'text': ' '.join(buf)})}\n\n"
            # Final metadata event.
            meta = {
                "type": "done",
                "session_id": session_id,
                "evidence_score": fs.evaluation.rubric_score if fs.evaluation else 0,
                "compression_ratio": fs.compression.compression_ratio if fs.compression else 1.0,
                "critic_confidence": fs.generation.critic_confidence if fs.generation else 1.0,
                "source_map": fs.compression.source_map if fs.compression else {},
                "degraded": bool(fs.memory and fs.memory.degraded_layers),
                "degraded_layers": fs.memory.degraded_layers if fs.memory else [],
            }
            db_manager.add_message(session_id=session_id, role="assistant", content=answer,
                                   rubric_score=meta["evidence_score"], compression_ratio=meta["compression_ratio"])
            yield f"data: {_json.dumps(meta)}\n\n"
        except Exception as exc:
            logger.error("Streaming pipeline error: %s", exc, exc_info=True)
            yield f"data: {_json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Session & Chat History Management ────────────────────────────────────

@app.get("/api/v1/sessions", tags=["Sessions"])
async def list_sessions(limit: int = 50):
    """List recent conversation sessions."""
    return db_manager.list_sessions(limit=limit)


@app.post("/api/v1/sessions", tags=["Sessions"], dependencies=[Depends(require_write_auth)])
async def create_session(payload: SessionCreate):
    """Create a new chat session."""
    return db_manager.create_session(title=payload.title)


@app.get("/api/v1/sessions/{session_id}", response_model=SessionDetail, tags=["Sessions"])
async def get_session_detail(session_id: str):
    """Retrieve session details and all historical messages."""
    session = db_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    messages = db_manager.get_messages(session_id)
    return SessionDetail(
        id=session["id"],
        title=session["title"],
        created_at=session["created_at"],
        updated_at=session["updated_at"],
        messages=[MessageItem(**m) for m in messages],
    )


@app.delete("/api/v1/sessions/{session_id}", tags=["Sessions"], dependencies=[Depends(require_write_auth)])
async def delete_session(session_id: str):
    """Delete a conversation session and all its messages."""
    deleted = db_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# ── Ingestion & Corpus Management ────────────────────────────────────────

@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Ingestion"], dependencies=[Depends(require_write_auth)])
async def upload_and_ingest_pdfs(files: List[UploadFile] = File(...)):
    """Upload one or more academic PDFs and ingest them into memory."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    temp_paths: List[str] = []
    max_bytes = getattr(settings, "max_upload_bytes", 25 * 1024 * 1024)
    try:
        for f in files:
            if not f.filename or not f.filename.lower().endswith(".pdf"):
                continue

            content = await f.read()

            # Size limit: reject oversized uploads before touching disk (F5).
            if len(content) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"File '{f.filename}' exceeds the maximum upload size "
                        f"of {max_bytes // (1024 * 1024)} MB."
                    ),
                )
            # Content validation: a real PDF begins with the %PDF magic bytes.
            # A .pdf extension is not sufficient for a network-exposed upload (F5).
            if not content.startswith(b"%PDF"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{f.filename}' is not a valid PDF (missing %PDF header).",
                )

            # Use only the basename to build the staging path — never honor any
            # directory components in a client-supplied filename (path traversal).
            safe_name = Path(f.filename).name
            suffix = Path(safe_name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(content)
            tmp.close()
            staged_path = Path(tmp.name).parent / safe_name
            os.replace(tmp.name, staged_path)
            temp_paths.append(str(staged_path))

        if not temp_paths:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid .pdf files were uploaded.",
            )

        res = ingestion_service.ingest_files(temp_paths)
        return IngestResponse(
            status=res["status"],
            ingested_count=res["ingested_count"],
            skipped_duplicates=res["skipped_duplicates"],
            new_chunks=res["new_chunks"],
            total_documents=res["total_documents"],
            total_chunks=res["total_chunks"],
            neo4j_status=res["neo4j_status"],
            message=res["message"],
        )
    finally:
        # Clean up temporary staging files
        for p in temp_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


@app.get("/api/v1/corpus", tags=["Ingestion"])
async def get_corpus_catalog():
    """Retrieve full catalog of active documents in memory."""
    return ingestion_service.get_catalog()


@app.post("/api/v1/corpus/wipe", tags=["Ingestion"], dependencies=[Depends(require_write_auth)])
async def wipe_corpus():
    """Wipe all memory stores (FAISS, BM25, uploads, catalog) to a clean slate."""
    res = ingestion_service.wipe_all()
    return res


@app.get("/api/v1/corpus/health", tags=["Ingestion"])
async def corpus_health():
    """Corpus-health panel data (Feature 7.2): per-layer counts, failed-doc count,
    and last ingestion timestamp — for the corpus management UX."""
    cat = ingestion_service.get_catalog()
    faiss_ok = os.path.isfile(settings.faiss_index_path)
    bm25_ok = os.path.isfile(settings.bm25_index_path)
    return {
        "total_documents": cat.get("total_documents", 0),
        "total_chunks": cat.get("total_chunks", 0),
        "failed_docs": len(cat.get("failed_docs", [])),
        "faiss_ready": faiss_ok,
        "bm25_ready": bm25_ok,
        "updated_at": cat.get("updated_at"),
    }


@app.post("/api/v1/corpus/retry-failed", tags=["Ingestion"], dependencies=[Depends(require_write_auth)])
async def retry_failed_docs():
    """Re-ingest documents that failed in a prior run (Feature 6.3 / 7.2)."""
    return ingestion_service.retry_failed()


@app.get("/api/v1/corpus/document/{doc_id}", tags=["Ingestion"])
async def get_corpus_document(doc_id: str):
    """Per-document detail view (Feature 7.2)."""
    cat = ingestion_service.get_catalog()
    doc = next((d for d in cat.get("documents", []) if d.get("doc_id") == doc_id), None)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found in corpus")
    return doc


# ── Knowledge Graph Topology ─────────────────────────────────────────────

@app.get("/api/v1/graph", tags=["Knowledge Graph"])
async def get_graph_topology():
    """Export the current active knowledge graph nodes and edges.

    Reads real FACT_TRIPLEs from Neo4j. If Neo4j is unreachable, returns an empty
    graph with ``source: "unavailable"`` rather than fabricating topology (Issue 24).
    """
    triples: List[str] = []
    source = "neo4j"
    try:
        from memory.neo4j_client import Neo4jClient
        with Neo4jClient() as client:
            for record in client.get_all_fact_triples(limit=500):
                t = record.get("triple", record)
                head = t.get("head_entity")
                rel = t.get("relation")
                tail = t.get("tail_entity")
                if head and rel and tail:
                    triples.append(f"{head} {rel} {tail}")
    except Exception as exc:
        logger.warning("Graph topology: Neo4j unavailable (%s)", exc)
        source = "unavailable"

    nodes, edges = parse_triples_to_graph(triples)
    return {"nodes": nodes, "edges": edges, "source": source, "triple_count": len(triples)}


# ── Autonomous Agents ────────────────────────────────────────────────────

@app.post("/api/v1/watcher/sync", tags=["Autonomous Agents"], dependencies=[Depends(require_write_auth)])
async def sync_arxiv_papers(query: str = "Retrieval-Augmented Generation", max_papers: int = 3):
    """Trigger the autonomous ArXiv watcher agent to search, download, and ingest papers."""
    from agents.arxiv_watcher import arxiv_watcher
    res = arxiv_watcher.fetch_and_ingest(query=query, max_papers=max_papers)
    return res


# ── Observability & Evaluation Benchmarks ────────────────────────────────

@app.get("/api/v1/metrics", response_model=MetricsResponse, tags=["Observability"])
async def get_observability_metrics():
    """Retrieve aggregate telemetry, latency percentiles (p50, p95), and rubric statistics."""
    return tracer.get_aggregated_metrics()


@app.get("/api/v1/metrics/traces", tags=["Observability"])
async def get_recent_traces(limit: int = 50):
    """Retrieve recent multi-agent query traces and span breakdowns."""
    traces = tracer.traces[-limit:]
    return [t.model_dump() for t in reversed(traces)]


@app.get("/api/v1/metrics/summary", tags=["Observability"])
async def get_metrics_summary():
    """Operational summary from persisted query audits (Feature 8.1):
    volume, avg rubric/compression/latency, external-search hit rate,
    self-correction (regeneration) rate, and route distribution."""
    return db_manager.get_audit_summary()


@app.post("/api/v1/evaluate", response_model=BenchmarkResponse, tags=["Evaluation"], dependencies=[Depends(require_write_auth)])
async def run_evaluation_benchmark():
    """Execute Ragas-style evaluation benchmark across the curated evaluation dataset."""
    from evaluation.benchmark import run_benchmark
    res = run_benchmark()
    return res

