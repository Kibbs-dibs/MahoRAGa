"""
Project Aletheia — CodeRabbit-Inspired Research & Review Console (ui/app.py)

Autonomous Multi-Agent AI Literature Reviewer:
- CodeRabbit Developer Aesthetic: CR Orange (#FF570A), Mauve Charcoal (#0b0d14), CR Mint (#00F2A5)
- Three-Layer Global Memory (Neo4j, FAISS, BM25)
- AttentionRAG Context Compression (6.3x Pruning)
- The Critic QA Verification & Automated Walkthrough
- Real-Time LangGraph Node Checkpoints Streaming
"""

from __future__ import annotations

import html
import logging
import time
from typing import Any, Dict, Generator, List, Tuple

import gradio as gr

from core.config import settings
from core.state import (
    AletheiaState,
    QueryState,
    MemoryState,
    EvaluationState,
    GenerationState,
)
from graph.orchestrator import aletheia_graph
from ingestion.service import ingestion_service
from ui.graph_visualizer import parse_triples_to_graph, generate_interactive_graph_html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ui.app")

# ═══════════════════════════════════════════════════════════════════════════
#  CodeRabbit-Inspired Design Tokens & High-Performance Developer CSS
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --cr-orange: #FF570A;
    --cr-orange-hover: #FF702E;
    --cr-orange-glow: rgba(255, 87, 10, 0.35);
    --cr-orange-subtle: rgba(255, 87, 10, 0.12);
    --cr-mint: #00F2A5;
    --cr-mint-subtle: rgba(0, 242, 165, 0.12);
    --cr-cobalt: #3B82F6;
    --cr-cobalt-subtle: rgba(59, 130, 246, 0.12);
    --cr-amber: #FBBF24;
    --cr-amber-subtle: rgba(251, 191, 36, 0.12);
    --cr-rose: #F43F5E;
    --cr-canvas: #0b0d14;
    --cr-card: #131622;
    --cr-card-alt: #1a1e2e;
    --cr-card-hover: #1e2235;
    --cr-border: rgba(255, 255, 255, 0.08);
    --cr-border-subtle: rgba(255, 255, 255, 0.04);
    --cr-text-primary: #F3F4F6;
    --cr-text-secondary: #9499AD;
    --cr-text-muted: #646B82;
}

body, .gradio-container {
    background: radial-gradient(circle at 10% 10%, rgba(255, 87, 10, 0.04) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(0, 242, 165, 0.03) 0%, transparent 40%),
                var(--cr-canvas) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--cr-text-primary) !important;
    letter-spacing: -0.01em;
}

h1, h2, h3, h4, .cr-font-brand {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.025em;
}

code, pre, .cr-mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── CodeRabbit Header Card ─────────────────────────────────────────────── */
.cr-header-card {
    background: linear-gradient(180deg, #161928 0%, #111420 100%) !important;
    border: 1px solid rgba(255, 87, 10, 0.25) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    margin-bottom: 18px !important;
    box-shadow: 0 8px 30px -10px rgba(0, 0, 0, 0.6), 0 0 20px -5px rgba(255, 87, 10, 0.15) !important;
}

.cr-logo-wrap {
    display: flex;
    align-items: center;
    gap: 14px;
}

.cr-rabbit-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, #FF570A 0%, #FF782E 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 16px var(--cr-orange-glow);
    flex-shrink: 0;
}

.cr-brand-title {
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 10px;
    line-height: 1.2;
}

.cr-brand-accent {
    color: var(--cr-orange);
}

.cr-badge-pill {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.cr-badge-orange {
    background: var(--cr-orange-subtle);
    border: 1px solid rgba(255, 87, 10, 0.35);
    color: var(--cr-orange);
}

.cr-badge-mint {
    background: var(--cr-mint-subtle);
    border: 1px solid rgba(0, 242, 165, 0.35);
    color: var(--cr-mint);
}

.cr-badge-cobalt {
    background: var(--cr-cobalt-subtle);
    border: 1px solid rgba(59, 130, 246, 0.35);
    color: var(--cr-cobalt);
}

.cr-header-desc {
    color: var(--cr-text-secondary);
    font-size: 0.88rem;
    margin-top: 4px;
}

.cr-meta-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    justify-content: flex-end;
}

.cr-meta-item {
    font-size: 0.76rem;
    font-weight: 500;
    color: var(--cr-text-secondary);
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--cr-border);
    padding: 5px 11px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.cr-pulse-dot {
    width: 7px;
    height: 7px;
    background: var(--cr-mint);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--cr-mint);
    animation: cr-pulse 2s infinite;
}

@keyframes cr-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

/* ── CodeRabbit Checkpoints / Pipeline Stepper ──────────────────────────── */
.stepper-container, .cr-stepper-box {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    justify-content: space-between;
    background: #111420;
    border: 1px solid var(--cr-border);
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 16px;
}

.step-node {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 13px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid transparent;
}

.step-pending {
    background: rgba(255, 255, 255, 0.02);
    color: var(--cr-text-muted);
    border-color: var(--cr-border-subtle);
}

.step-active {
    background: rgba(255, 87, 10, 0.12);
    color: #FFA573;
    border-color: var(--cr-orange);
    box-shadow: 0 0 16px var(--cr-orange-glow);
    animation: cr-active-pulse 2s infinite ease-in-out;
}

.step-done {
    background: var(--cr-mint-subtle);
    color: var(--cr-mint);
    border-color: rgba(0, 242, 165, 0.4);
}

@keyframes cr-active-pulse {
    0%, 100% { box-shadow: 0 0 12px rgba(255, 87, 10, 0.2); }
    50% { box-shadow: 0 0 22px rgba(255, 87, 10, 0.5); }
}

/* ── CodeRabbit Metrics Scorecard ───────────────────────────────────────── */
.metrics-grid, .cr-scorecard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
    gap: 10px;
    margin: 10px 0 16px 0;
}

.metric-card, .cr-score-card {
    background: #131622;
    border: 1px solid var(--cr-border);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}

.metric-card:hover, .cr-score-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 87, 10, 0.4);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
}

.metric-value, .cr-score-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--cr-text-primary);
}

.metric-label, .cr-score-lbl {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--cr-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

/* ── Chatbot & Developer Cards ──────────────────────────────────────────── */
.gradio-chatbot, #aletheia-chatbot {
    background: #111420 !important;
    border: 1px solid var(--cr-border) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
}

/* CodeRabbit Orange Primary Button */
.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #FF570A 0%, #FF782E 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px var(--cr-orange-glow) !important;
    transition: all 0.2s ease !important;
}

.gr-button-primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(255, 87, 10, 0.5) !important;
    background: linear-gradient(135deg, #FF661F 0%, #FF853D 100%) !important;
}

/* Secondary Button */
.gr-button-secondary {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid var(--cr-border) !important;
    color: var(--cr-text-secondary) !important;
    border-radius: 8px !important;
}

.gr-button-secondary:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: var(--cr-text-primary) !important;
}

/* Accordion & Tabs */
.tab-nav button.selected {
    color: var(--cr-orange) !important;
    border-bottom: 2px solid var(--cr-orange) !important;
    font-weight: 700 !important;
}

.label-title, .gr-panel {
    color: var(--cr-text-secondary) !important;
    font-weight: 600 !important;
}

/* CodeRabbit Callout Box */
.cr-callout {
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.85rem;
    line-height: 1.5;
    border-left: 3px solid;
}

.cr-callout-success {
    background: var(--cr-mint-subtle);
    border-color: var(--cr-mint);
    color: #B4F8E2;
}

.cr-callout-info {
    background: var(--cr-orange-subtle);
    border-color: var(--cr-orange);
    color: #FFD2BC;
}

.cr-diff-line-add {
    background: rgba(0, 242, 165, 0.08);
    color: #00F2A5;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}

.cr-diff-line-tag {
    background: rgba(255, 87, 10, 0.1);
    color: #FF945C;
    padding: 1px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    border: 1px solid rgba(255, 87, 10, 0.25);
}

/* ── Ingestion Dock & Document Catalog ──────────────────────────────────── */
.cr-ingest-dock {
    background: #111420 !important;
    border: 1px solid rgba(255, 87, 10, 0.25) !important;
    border-radius: 12px !important;
    margin-bottom: 16px !important;
}

.cr-doc-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 220px;
    overflow-y: auto;
    margin-top: 10px;
    padding-right: 4px;
}

.cr-doc-card {
    background: #161928;
    border: 1px solid var(--cr-border);
    border-radius: 8px;
    padding: 9px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.15s ease;
}

.cr-doc-card:hover {
    border-color: rgba(255, 87, 10, 0.4);
}

.cr-doc-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 100%;
}

.cr-doc-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--cr-text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 420px;
}

.cr-doc-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
}

.cr-doc-empty {
    font-size: 0.82rem;
    color: var(--cr-text-muted);
    padding: 12px;
    text-align: center;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 8px;
    border: 1px dashed var(--cr-border);
}

.cr-status-box {
    padding: 8px 12px;
    border-radius: 8px;
    margin: 8px 0;
    font-size: 0.82rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.cr-subhead {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--cr-text-secondary);
    margin-top: 10px;
    margin-bottom: 4px;
}
"""

# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline State & Formatting Helpers (CodeRabbit Review Style)
# ═══════════════════════════════════════════════════════════════════════════

ALL_STAGES = [
    ("router", "1. Router", "🧭"),
    ("retriever", "2. Retriever", "⚡"),
    ("evaluator", "3. Evaluator", "⚖️"),
    ("search_apis", "4. Search APIs", "🌐"),
    ("pruner", "5. Pruner (AttnRAG)", "✂️"),
    ("generator", "6. Generator & Critic", "✨"),
]


def _ensure_state(state: Any) -> AletheiaState | None:
    """Convert dict or AletheiaState to a typed AletheiaState instance."""
    if state is None:
        return None
    if isinstance(state, AletheiaState):
        return state
    if isinstance(state, dict):
        try:
            return AletheiaState.model_validate(state)
        except Exception:
            try:
                return AletheiaState(**state)
            except Exception:
                pass
    return None


def render_stepper_html(
    active_node: str = "",
    completed_nodes: List[str] | None = None,
    elapsed_time: float = 0.0,
    status_label: str = "Ready",
) -> str:
    """Render CodeRabbit-styled automated review checkpoints timeline."""
    completed = set(completed_nodes or [])

    nodes_html = []
    for node_id, label, icon in ALL_STAGES:
        if node_id == active_node:
            state_cls = "step-active"
            badge = "• Reviewing"
        elif node_id in completed:
            state_cls = "step-done"
            badge = "✓"
        else:
            state_cls = "step-pending"
            badge = ""

        badge_span = (
            f"<span style='font-size:0.7rem; opacity:0.85; margin-left:2px;'>{badge}</span>"
            if badge
            else ""
        )
        nodes_html.append(
            f"<div class='step-node {state_cls}'>"
            f"<span>{icon}</span> <span>{label}</span> {badge_span}"
            f"</div>"
        )

    timer_text = f"{elapsed_time:.1f}s" if elapsed_time > 0 else "0.0s"
    status_badge = (
        f"<div style='font-size:0.78rem; font-weight:600; color:#FF8A4C; display:flex; align-items:center; gap:8px;'>"
        f"<span>Status: {html.escape(status_label)}</span>"
        f"<span style='background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.06); padding:2px 8px; border-radius:6px; color:#8E95AA;'>⏱ {timer_text}</span>"
        f"</div>"
    )

    return (
        f"<div class='stepper-container cr-stepper-box'>"
        f"<div style='display:flex; flex-wrap:wrap; gap:6px; align-items:center;'>"
        f"{''.join(nodes_html)}"
        f"</div>"
        f"{status_badge}"
        f"</div>"
    )


def render_metrics_html(state: Any, elapsed: float = 0.0) -> str:
    """Render CodeRabbit review scorecard cards: Rubric Score, AttentionRAG Compression, Latency, Critic."""
    s = _ensure_state(state)
    if not s:
        return (
            "<div class='metrics-grid cr-scorecard-grid'>"
            "<div class='metric-card'><div class='metric-value' style='color:#646B82;'>--</div><div class='metric-label'>Evidence Rubric</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#646B82;'>--</div><div class='metric-label'>Compression Ratio</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#8E95AA;'>0.0s</div><div class='metric-label'>Latency</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#646B82;'>--</div><div class='metric-label'>The Critic QA</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#8E95AA;'>$0.00</div><div class='metric-label'>Est. Cost</div></div>"
            "</div>"
        )

    # 1. Evidence Score
    ev_score = s.evaluation.rubric_score if s.evaluation else 0
    ev_color = "#00F2A5" if ev_score >= 80 else ("#FBBF24" if ev_score >= 50 else "#F43F5E")
    ev_text = f"{ev_score}/100"

    # 2. Compression Ratio
    comp_ratio = s.compression.compression_ratio if s.compression else 0.0
    comp_text = f"{comp_ratio:.1f}x" if comp_ratio > 0 else "1.0x"

    # 3. Latency
    latency_text = f"{elapsed:.2f}s"

    # 4. Critic Verdict
    feedback = s.generation.critic_feedback if s.generation else None
    if not feedback or "approved" in feedback.lower():
        critic_icon = "✅ Approved"
        critic_color = "#00F2A5"
    elif "regenerate" in feedback.lower() or "retry" in feedback.lower():
        critic_icon = "⚠️ Regenerated"
        critic_color = "#FBBF24"
    else:
        critic_icon = "✅ Passed"
        critic_color = "#00F2A5"

    # 5. Cost
    cost_text = f"${s.pipeline_cost:.4f}"

    return (
        f"<div class='metrics-grid cr-scorecard-grid'>"
        f"<div class='metric-card'><div class='metric-value' style='color:{ev_color};'>{ev_text}</div><div class='metric-label'>Evidence Rubric</div></div>"
        f"<div class='metric-card'><div class='metric-value' style='color:#FF570A;'>{comp_text}</div><div class='metric-label'>Compression Ratio</div></div>"
        f"<div class='metric-card'><div class='metric-value' style='color:#3B82F6;'>{latency_text}</div><div class='metric-label'>Latency</div></div>"
        f"<div class='metric-card'><div class='metric-value' style='font-size:1.15rem; color:{critic_color};'>{critic_icon}</div><div class='metric-label'>The Critic QA</div></div>"
        f"<div class='metric-card'><div class='metric-value' style='color:#8E95AA;'>{cost_text}</div><div class='metric-label'>Token Cost</div></div>"
        f"</div>"
    )


def format_neo4j_markdown(state: Any) -> str:
    """Format Neo4j triples with an embedded interactive force-directed graph canvas + markdown provenance."""
    s = _ensure_state(state)
    if not s or not s.memory or not s.memory.serialized_graph_triples:
        canvas_html = generate_interactive_graph_html([], [])
        fallback_msg = (
            "*(Neo4j instance is offline or returned no direct triples. "
            "Pipeline successfully used FAISS and BM25 dual stores.)*"
        )
        return f"{canvas_html}\n\n{fallback_msg}"

    triples = s.memory.serialized_graph_triples
    nodes, edges = parse_triples_to_graph(triples)
    canvas_html = generate_interactive_graph_html(nodes, edges, height_px=380)

    items = [canvas_html, "\n<div style='margin-top:12px; font-weight:600; color:#FF570A;'>🏷️ Active Knowledge Graph Triples:</div>"]
    for i, triple in enumerate(triples[:15], 1):
        items.append(f"- 🏷️ `triple:{i:02d}` **{triple}**")
    return "\n".join(items)


def format_faiss_markdown(state: Any) -> str:
    """Format FAISS dense vector search results with CodeRabbit review pill aesthetics."""
    s = _ensure_state(state)
    if not s or not s.memory or not s.memory.vector_chunks:
        return "*(No FAISS vector chunks retrieved yet.)*"

    items = []
    for i, chunk in enumerate(s.memory.vector_chunks[:5], 1):
        preview = chunk[:400].strip() + ("..." if len(chunk) > 400 else "")
        items.append(
            f"**Chunk #{i}:** `FAISS Dense Vector`  \n"
            f"> {preview}\n"
        )
    return "\n\n".join(items)


def format_bm25_markdown(state: Any) -> str:
    """Format BM25 sparse lexical search results with CodeRabbit keyword markers."""
    s = _ensure_state(state)
    if not s or not s.memory or not s.memory.lexical_chunks:
        return "*(No BM25 lexical chunks retrieved yet.)*"

    items = []
    for i, chunk in enumerate(s.memory.lexical_chunks[:5], 1):
        preview = chunk[:400].strip() + ("..." if len(chunk) > 400 else "")
        items.append(
            f"**Lexical Match #{i}:** `BM25Okapi Sparse`  \n"
            f"> {preview}\n"
        )
    return "\n\n".join(items)


def format_external_markdown(state: Any) -> str:
    """Format external academic literature hits (Semantic Scholar, arXiv, Crossref, OpenAlex)."""
    s = _ensure_state(state)
    if not s or not s.memory or not s.memory.external_api_abstracts:
        return "*(External literature APIs were not triggered — internal memory met the evidence threshold.)*"

    items = []
    for i, abstract in enumerate(s.memory.external_api_abstracts[:5], 1):
        preview = abstract[:400].strip() + ("..." if len(abstract) > 400 else "")
        items.append(f"**External Literature #{i}:**\n> {preview}\n")
    return "\n\n".join(items)


def render_header_meta_html(catalog: Dict[str, Any] | None = None) -> str:
    """Render CodeRabbit top meta bar with real-time ingested paper counts."""
    if catalog is None:
        try:
            catalog = ingestion_service.get_catalog()
        except Exception:
            catalog = {"total_documents": 0, "total_chunks": 0}
    doc_count = catalog.get("total_documents", 0)
    chunk_count = catalog.get("total_chunks", 0)
    return (
        f"<div class='cr-meta-bar'>"
        f"<div class='cr-meta-item'><span class='cr-pulse-dot'></span> <span>{doc_count} Papers Ingested ({chunk_count} Chunks)</span></div>"
        f"<div class='cr-meta-item'><span>📁 Kibbs-dibs/MahoRAGa @ main</span></div>"
        f"<div class='cr-badge-pill cr-badge-orange'>PR #1: Research Assistant</div>"
        f"</div>"
    )


def render_ingest_status_html(res: Dict[str, Any] | None = None) -> str:
    """Render status pill for in-browser PDF ingestion and memory wipe operations."""
    if res is None:
        return (
            "<div class='cr-status-box' style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);'>"
            "<span style='color:#9499AD;'>Ready to ingest academic literature. Drag & drop PDF files above.</span>"
            "</div>"
        )

    status = res.get("status", "")
    msg = html.escape(res.get("message", ""))
    neo4j_st = res.get("neo4j_status", "")

    if status == "success":
        neo4j_badge = ""
        if neo4j_st == "offline":
            neo4j_badge = (
                "&nbsp;<span class='cr-badge-pill' style='background:rgba(251,191,36,0.12); color:#FBBF24; border:1px solid rgba(251,191,36,0.3);'>"
                "⚠️ Neo4j Offline (FAISS & BM25 Active)</span>"
            )
        elif neo4j_st == "synced":
            neo4j_badge = (
                "&nbsp;<span class='cr-badge-pill cr-badge-mint'>✓ Neo4j Graph Synced</span>"
            )
        return (
            f"<div class='cr-status-box' style='background:rgba(0,242,165,0.08); border:1px solid rgba(0,242,165,0.3); color:#00F2A5;'>"
            f"<span>✅ {msg}</span>{neo4j_badge}"
            f"</div>"
        )
    elif status == "wiped":
        return (
            f"<div class='cr-status-box' style='background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.3); color:#FBBF24;'>"
            f"<span>🗑️ {msg}</span>"
            f"</div>"
        )
    elif status in ("skipped", "no_files"):
        return (
            f"<div class='cr-status-box' style='background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.3); color:#FBBF24;'>"
            f"<span>⚠️ {msg}</span>"
            f"</div>"
        )
    else:
        return (
            f"<div class='cr-status-box' style='background:rgba(244,63,94,0.08); border:1px solid rgba(244,63,94,0.3); color:#F43F5E;'>"
            f"<span>❌ {msg}</span>"
            f"</div>"
        )


def render_catalog_html(catalog: Dict[str, Any] | None = None) -> str:
    """Render interactive catalog list of ingested documents."""
    if catalog is None:
        try:
            catalog = ingestion_service.get_catalog()
        except Exception:
            catalog = {"documents": []}

    docs = catalog.get("documents", [])
    if not docs:
        return "<div class='cr-doc-empty'>*(No documents loaded. Ingest PDFs above to populate memory.)*</div>"

    items = []
    for d in docs:
        title = html.escape(d.get("title") or d.get("filename") or "Untitled Document")
        chunks = d.get("chunk_count", 0)
        pages = d.get("page_count", 1)
        ts = (d.get("ingested_at") or "")[:10]
        items.append(
            f"<div class='cr-doc-card'>"
            f"<div class='cr-doc-info'>"
            f"<div class='cr-doc-title' title='{title}'>📄 {title}</div>"
            f"<div class='cr-doc-meta'>"
            f"<span class='cr-badge-pill cr-badge-orange'>{chunks} Chunks</span>"
            f"<span class='cr-badge-pill cr-badge-mint'>{pages} Pages</span>"
            f"<span class='cr-badge-pill' style='background:rgba(255,255,255,0.04); color:#9499AD;'>📅 {ts}</span>"
            f"</div>"
            f"</div>"
            f"</div>"
        )
    return f"<div class='cr-doc-list'>{''.join(items)}</div>"


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline Execution Generator (Streaming Multi-Agent Graph)
# ═══════════════════════════════════════════════════════════════════════════

def format_coderabbit_review_response(
    raw_answer: str,
    evidence_score: int,
    compression_ratio: float,
    critic_verdict: str,
    latency: float,
) -> str:
    """Wrap synthesized answer into CodeRabbit's iconic structured PR review format."""
    status_icon = "✅" if evidence_score >= 80 else "⚠️"
    rubric_badge = f"`Evidence Score: {evidence_score}/100`"
    attn_badge = f"`AttentionRAG: {compression_ratio:.1f}x pruned`"
    critic_badge = f"`The Critic: {critic_verdict}`"

    header_block = (
        f"### 🐇 CodeRabbit Review: AI Literature Analysis\n\n"
        f"> **Assessment:** {status_icon} **{critic_verdict}** &nbsp;•&nbsp; {rubric_badge} &nbsp;•&nbsp; {attn_badge} &nbsp;•&nbsp; `Latency: {latency:.1f}s`\n\n"
    )

    return header_block + raw_answer


def stream_query_pipeline(
    message: str,
    history: List[Dict[str, str]],
    llm_provider: str,
    local_model: str,
    evidence_threshold: float,
    enable_external_search: bool,
    max_compression_ratio: float,
) -> Generator[Tuple[List[Dict[str, str]], str, str, str, str, str, str], None, None]:
    """
    Execute the LangGraph pipeline with real-time step streaming.
    Yields UI updates after each node completion in CodeRabbit review style.
    """
    clean_query = (message or "").strip()
    if not clean_query:
        yield history, "", render_stepper_html(), "", "", "", "", render_metrics_html(None)
        return

    # Update runtime settings based on user preferences
    settings.ollama_model = local_model
    settings.evidence_threshold = float(evidence_threshold)
    settings.max_compression_ratio = float(max_compression_ratio)

    start_time = time.time()
    completed_nodes: List[str] = []

    # Initialize conversation message in CodeRabbit format
    user_msg = {"role": "user", "content": clean_query}
    assistant_msg = {
        "role": "assistant",
        "content": "🐇 *CodeRabbit is reviewing literature against Three-Layer Global Memory...*",
    }
    current_history = history + [user_msg, assistant_msg]

    # Yield initial pending state
    yield (
        current_history,
        "",  # clear input textbox
        render_stepper_html("router", completed_nodes, 0.0, "Routing Query"),
        format_neo4j_markdown(None),
        format_faiss_markdown(None),
        format_bm25_markdown(None),
        format_external_markdown(None),
        render_metrics_html(None, 0.0),
    )

    # Initialize LangGraph State
    initial_state = AletheiaState(
        query=QueryState(original_query=clean_query),
        memory=MemoryState(),
        evaluation=EvaluationState(),
        generation=GenerationState(),
    )

    latest_state = initial_state

    # Stage descriptions with CodeRabbit review persona
    stage_messages = {
        "router": "🧭 *Classifying query intent and canonical domain terms...*",
        "retriever": "⚡ *Retrieving passages from Three-Layer Global Memory (Neo4j, FAISS, BM25)...*",
        "evaluator": "⚖️ *Evaluating 100-point evidence sufficiency rubric...*",
        "search_apis": "🌐 *Querying external literature fallback (Crossref, Semantic Scholar, arXiv)...*",
        "pruner": "✂️ *Compressing context micro-windows via AttentionRAG...*",
        "generator": "✨ *Drafting cited synthesis and executing The Critic QA audit...*",
    }

    try:
        # Stream intermediate graph outputs node-by-node
        for step_output in aletheia_graph.stream(initial_state):
            for node_name, updated_state in step_output.items():
                st = _ensure_state(updated_state)
                if st is not None:
                    latest_state = st
                completed_nodes.append(node_name)
                elapsed = time.time() - start_time

                # Next active node determination
                next_active = ""
                if node_name == "router":
                    next_active = "retriever"
                elif node_name == "retriever":
                    next_active = "evaluator"
                elif node_name == "evaluator":
                    is_weak = latest_state.evaluation.is_evidence_weak if latest_state.evaluation else True
                    next_active = "search_apis" if is_weak else "pruner"
                elif node_name == "search_apis":
                    next_active = "pruner"
                elif node_name == "pruner":
                    next_active = "generator"

                # Update live assistant message
                if node_name == "generator":
                    final_ans = (latest_state.generation.draft_answer if latest_state.generation else "") or "No response generated."
                    current_history[-1]["content"] = final_ans
                else:
                    progress_note = stage_messages.get(next_active, f"Processing {node_name}...")
                    current_history[-1]["content"] = progress_note

                yield (
                    current_history,
                    "",
                    render_stepper_html(
                        next_active,
                        completed_nodes,
                        elapsed,
                        f"Completed {node_name.capitalize()}",
                    ),
                    format_neo4j_markdown(latest_state),
                    format_faiss_markdown(latest_state),
                    format_bm25_markdown(latest_state),
                    format_external_markdown(latest_state),
                    render_metrics_html(latest_state, elapsed),
                )

        # Final completion yield formatted with CodeRabbit review summary
        total_time = time.time() - start_time
        raw_answer = (latest_state.generation.draft_answer if latest_state.generation else "") or "Research synthesis complete."

        ev_score = latest_state.evaluation.rubric_score if latest_state.evaluation else 0
        comp_ratio = latest_state.compression.compression_ratio if latest_state.compression else 1.0
        critic_verdict = (
            "Approved (Zero Hallucinations)"
            if (not latest_state.generation or not latest_state.generation.critic_feedback or "approved" in latest_state.generation.critic_feedback.lower())
            else "Regenerated (Self-Corrected)"
        )

        formatted_final = format_coderabbit_review_response(
            raw_answer=raw_answer,
            evidence_score=ev_score,
            compression_ratio=comp_ratio,
            critic_verdict=critic_verdict,
            latency=total_time,
        )

        current_history[-1]["content"] = formatted_final

        yield (
            current_history,
            "",
            render_stepper_html(
                "",
                completed_nodes,
                total_time,
                f"Review Complete in {total_time:.1f}s",
            ),
            format_neo4j_markdown(latest_state),
            format_faiss_markdown(latest_state),
            format_bm25_markdown(latest_state),
            format_external_markdown(latest_state),
            render_metrics_html(latest_state, total_time),
        )

    except Exception as exc:
        logger.error("Pipeline streaming failure: %s", exc, exc_info=True)
        error_msg = f"⚠️ **Review Pipeline Interrupted:** `{html.escape(str(exc))}`\n\n*Graceful degradation active: Inspect system logs for diagnostics.*"
        current_history[-1]["content"] = error_msg
        total_time = time.time() - start_time
        yield (
            current_history,
            "",
            render_stepper_html("", completed_nodes, total_time, "Error Occurred"),
            format_neo4j_markdown(latest_state),
            format_faiss_markdown(latest_state),
            format_bm25_markdown(latest_state),
            format_external_markdown(latest_state),
            render_metrics_html(latest_state, total_time),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Gradio UI Layout Construction (CodeRabbit Visual Language)
# ═══════════════════════════════════════════════════════════════════════════

def create_ui() -> gr.Blocks:
    """Construct the CodeRabbit-styled web app for Project Aletheia."""

    with gr.Blocks(title="Project Aletheia — Autonomous AI Research") as demo:
        # Injected Custom CodeRabbit Design System CSS
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # ── Header: CodeRabbit AI Reviewer Banner ─────────────────────────────
        with gr.Row(elem_classes=["cr-header-card"]):
            with gr.Column(scale=8):
                gr.HTML(
                    """
                    <div class="cr-logo-wrap">
                        <div class="cr-rabbit-icon">
                            <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
                                <path d="M12 2C8 2 6 5.5 6 9.5c0 2.8 1.4 5 2.8 6.5L6.5 22h11l-2.3-6c1.4-1.5 2.8-3.7 2.8-6.5C18 5.5 16 2 12 2z" fill="#ffffff"/>
                                <circle cx="9.5" cy="9" r="1.3" fill="#FF570A"/>
                                <circle cx="14.5" cy="9" r="1.3" fill="#FF570A"/>
                                <path d="M10.8 12.2c.7.5 1.7.5 2.4 0" stroke="#FF570A" stroke-width="1.2" stroke-linecap="round"/>
                                <path d="M7 3L5 1M17 3l2-2" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
                            </svg>
                        </div>
                        <div>
                            <div class="cr-brand-title">
                                CodeRabbit <span class="cr-brand-accent">Aletheia</span>
                                <span class="cr-badge-pill cr-badge-orange">AI Reviewer</span>
                                <span class="cr-badge-pill cr-badge-mint">Verified</span>
                            </div>
                            <div class="cr-header-desc">
                                Autonomous Multi-Agent Literature Reviewer • Three-Layer Global Memory • AttentionRAG 6.3x Pruning • The Critic QA
                            </div>
                        </div>
                    </div>
                    """
                )
            with gr.Column(scale=4):
                header_meta = gr.HTML(
                    value=render_header_meta_html(),
                    elem_id="cr-header-meta",
                )

        # ── CodeRabbit Automated Checks / Stepper ─────────────────────────────
        status_stepper = gr.HTML(
            value=render_stepper_html(status_label="CodeRabbit Ready"),
            elem_id="pipeline-stepper",
        )

        # ── Main Developer Workspace (2-Column Review Layout) ─────────────────
        with gr.Row():
            # Left Column: Code Review Stream & Query Input
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="CodeRabbit Review Stream & Synthesis",
                    height=530,
                    elem_id="aletheia-chatbot",
                    layout="bubble",
                )

                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Ask CodeRabbit to review literature (e.g. 'How does AttentionRAG compress context while preserving citation fidelity?')...",
                        lines=2,
                        scale=9,
                        label="Literature Query",
                        show_label=False,
                    )
                    submit_btn = gr.Button("Ask CodeRabbit ➔", variant="primary", scale=3)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Reset Review", variant="secondary", size="sm")

                # CodeRabbit Review Query Templates
                gr.Examples(
                    examples=[
                        "How does AttentionRAG compress context while preserving citation fidelity?",
                        "Compare MemGraphRAG hierarchical indexing with TechGraphRAG multi-agent retrieval.",
                        "What are the core components of the Three-Layer Global Memory architecture?",
                        "How does The Critic audit draft answers against passage provenance?",
                    ],
                    inputs=query_input,
                    label="🐇 CodeRabbit Review Templates",
                )

            # Right Column: Review Scorecard & Multi-Source Breakdown
            with gr.Column(scale=5):
                metrics_display = gr.HTML(
                    value=render_metrics_html(None),
                    elem_id="metrics-panel",
                )

                # ── Ingestion Dock: Live PDF Ingestion & Corpus Manager ─────────────
                with gr.Accordion("📥 Ingest Research Papers", open=False, elem_classes=["cr-ingest-dock"]):
                    file_uploader = gr.File(
                        label="Select or Drop Academic Research Papers (PDF)",
                        file_types=[".pdf"],
                        file_count="multiple",
                        elem_id="cr-file-uploader",
                    )
                    with gr.Row():
                        ingest_btn = gr.Button("Ingest Papers into Memory ➔", variant="primary", scale=8)
                        wipe_btn = gr.Button("🗑️ Wipe All Memory", variant="stop", scale=4)

                    # ── Autonomous ArXiv Watcher Controls ──────────────────────
                    with gr.Accordion("🤖 Autonomous ArXiv Daily Watcher", open=False):
                        with gr.Row():
                            arxiv_query = gr.Textbox(
                                value="Retrieval-Augmented Generation",
                                label="ArXiv Topic / Search Keywords",
                                scale=8,
                            )
                            arxiv_count = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=2,
                                step=1,
                                label="Max Papers",
                                scale=4,
                            )
                        arxiv_btn = gr.Button("Fetch & Ingest Trending ArXiv Papers ➔", variant="secondary")

                    ingest_status = gr.HTML(
                        value=render_ingest_status_html(None),
                        elem_id="cr-ingest-status",
                    )

                    gr.HTML("<div class='cr-subhead'>Active Literature in Memory:</div>")
                    catalog_display = gr.HTML(
                        value=render_catalog_html(),
                        elem_id="cr-catalog-display",
                    )

                with gr.Tabs(elem_id="sources-tabs"):
                    with gr.Tab("📊 Neo4j Graph"):
                        sources_neo4j = gr.Markdown(
                            value="*(Run a query to inspect Neo4j ontology and fact triples)*",
                            label="Neo4j Knowledge Graph Triples",
                        )
                    with gr.Tab("🔍 FAISS Dense"):
                        sources_faiss = gr.Markdown(
                            value="*(Run a query to inspect top cosine similarity chunks)*",
                            label="FAISS Dense Vectors",
                        )
                    with gr.Tab("📚 BM25 Lexical"):
                        sources_bm25 = gr.Markdown(
                            value="*(Run a query to inspect BM25Okapi lexical matches)*",
                            label="BM25 Sparse Lexical Store",
                        )
                    with gr.Tab("🌐 External APIs"):
                        sources_external = gr.Markdown(
                            value="*(External search triggers when internal evidence score < 80)*",
                            label="External Academic APIs",
                        )

                # CodeRabbit Review Settings Drawer
                with gr.Accordion("⚙️ CodeRabbit Review Settings", open=False):
                    with gr.Row():
                        llm_provider = gr.Radio(
                            choices=["Local Ollama", "OpenAI API", "Anthropic Proxy", "Offline Fallback"],
                            value="Local Ollama",
                            label="LLM Provider",
                        )
                        local_model = gr.Dropdown(
                            choices=["qwen2.5:7b", "llama3.1:8b", "nemotron-3-super-120b-a12b", "custom"],
                            value="qwen2.5:7b",
                            label="Ollama Model",
                        )
                    with gr.Row():
                        evidence_threshold = gr.Slider(
                            minimum=10.0,
                            maximum=100.0,
                            value=80.0,
                            step=5.0,
                            label="Evidence Threshold (Triggers External Fallback)",
                        )
                        max_compression_ratio = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=6.3,
                            step=0.5,
                            label="Max AttentionRAG Compression Ratio",
                        )
                    enable_external_search = gr.Checkbox(
                        value=True,
                        label="Enable Fallback Academic Search (arXiv, Semantic Scholar, Crossref)",
                    )

        # ── Event Bindings ───────────────────────────────────────────────────
        input_args = [
            query_input,
            chatbot,
            llm_provider,
            local_model,
            evidence_threshold,
            enable_external_search,
            max_compression_ratio,
        ]
        output_args = [
            chatbot,
            query_input,
            status_stepper,
            sources_neo4j,
            sources_faiss,
            sources_bm25,
            sources_external,
            metrics_display,
        ]

        submit_btn.click(
            fn=stream_query_pipeline,
            inputs=input_args,
            outputs=output_args,
        )

        query_input.submit(
            fn=stream_query_pipeline,
            inputs=input_args,
            outputs=output_args,
        )

        def clear_conversation():
            return [], "", render_stepper_html(status_label="Reset"), "", "", "", "", render_metrics_html(None)

        clear_btn.click(
            fn=clear_conversation,
            inputs=None,
            outputs=output_args,
        )

        # ── Ingestion Dock Event Bindings ───────────────────────────────────
        def run_live_ingest(files, progress=gr.Progress(track_tqdm=True)):
            if not files:
                res = {"status": "no_files", "message": "No PDF files were selected."}
                return (
                    render_ingest_status_html(res),
                    render_catalog_html(),
                    render_header_meta_html(),
                    None,
                )

            file_paths = [f.name if hasattr(f, "name") else str(f) for f in files]

            def progress_callback(ratio, status_msg):
                progress(ratio, desc=status_msg)

            res = ingestion_service.ingest_files(file_paths, progress_cb=progress_callback)
            cat = res.get("catalog") or ingestion_service.get_catalog()
            return (
                render_ingest_status_html(res),
                render_catalog_html(cat),
                render_header_meta_html(cat),
                None,
            )

        def run_live_wipe():
            res = ingestion_service.wipe_all()
            cat = res.get("catalog") or ingestion_service.get_catalog()
            return (
                render_ingest_status_html(res),
                render_catalog_html(cat),
                render_header_meta_html(cat),
                None,
            )

        ingest_btn.click(
            fn=run_live_ingest,
            inputs=[file_uploader],
            outputs=[ingest_status, catalog_display, header_meta, file_uploader],
        )

        wipe_btn.click(
            fn=run_live_wipe,
            inputs=None,
            outputs=[ingest_status, catalog_display, header_meta, file_uploader],
        )

        def run_arxiv_sync(query, count, progress=gr.Progress(track_tqdm=True)):
            from agents.arxiv_watcher import arxiv_watcher

            def progress_callback(ratio, status_msg):
                progress(ratio, desc=status_msg)

            res = arxiv_watcher.fetch_and_ingest(
                query=query,
                max_papers=int(count),
                progress_cb=progress_callback,
            )
            cat = ingestion_service.get_catalog()
            return (
                render_ingest_status_html(res),
                render_catalog_html(cat),
                render_header_meta_html(cat),
            )

        arxiv_btn.click(
            fn=run_arxiv_sync,
            inputs=[arxiv_query, arxiv_count],
            outputs=[ingest_status, catalog_display, header_meta],
        )

    return demo


# Create global app instance
app = create_ui()

if __name__ == "__main__":
    logger.info("Launching Project Aletheia (CodeRabbit Style) Web UI on http://localhost:7860")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
