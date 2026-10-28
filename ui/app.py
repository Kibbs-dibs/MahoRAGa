"""
Project Aletheia — Reading-Room Research Console (ui/app.py)

Autonomous Multi-Agent AI Literature Reviewer:
- "Reading room" aesthetic: warm ink surfaces (#15191c), archival-paper text
  (#f1ece1), a scholarly oxblood accent (#a23c35) and muted brass rules
  (#b79a63); Spectral serif for reading, Inter for controls.
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
@import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Inter:wght@400;500;600&display=swap');

/* ═══════════════════════════════════════════════════════════════════════════
   ALETHEIA — "Reading Room" design system.
   A bound-journal palette (deep ink surfaces, warm archival paper text,
   a scholarly oxblood accent, muted brass rules) for a literature-research
   instrument. One reading serif (Spectral) carries synthesis + headings; a
   quiet sans (Inter) handles dense controls. No neon, no glass-card kit, no
   decorative monospace — structure is drawn with ink-weight rules, not shadows.
   ═══════════════════════════════════════════════════════════════════════════ */

:root {
    --ink:          #15191c;   /* darkest surface — warm slate ink, not tinted black */
    --ink-raised:   #1c2227;   /* raised panel */
    --ink-sunken:   #11161a;   /* reading well / chat canvas */
    --paper:        #f1ece1;   /* archival paper — primary text on dark */
    --paper-dim:    #cbc3b4;   /* secondary text */
    --paper-faint:  #8d8676;   /* tertiary / captions */
    --oxblood:      #a23c35;   /* the single accent — library-binding red */
    --oxblood-soft: #c65a4f;   /* lighter oxblood for text-on-dark legibility */
    --brass:        #b79a63;   /* muted brass — rules, marks, "done" states */
    --brass-dim:    rgba(183, 154, 99, 0.22);
    --rule:         rgba(183, 154, 99, 0.20);  /* brass hairline */
    --rule-faint:   rgba(241, 236, 225, 0.08);
    --serif: 'Spectral', Georgia, 'Times New Roman', serif;
    --sans:  'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --ok:   #7fa06a;   /* muted sage — verified (not acid green) */
    --warn: #c9972f;   /* ochre — caution */
    --bad:  #b6443c;   /* oxblood-adjacent — failure */
}

/* ── Canvas: a quiet reading-room gradient, no multi-point neon glows ─────── */
body, .gradio-container {
    background:
        radial-gradient(1200px 600px at 78% -12%, rgba(162, 60, 53, 0.08), transparent 55%),
        var(--ink) !important;
    background-attachment: fixed !important;
    color: var(--paper) !important;
    font-family: var(--sans) !important;
    letter-spacing: 0;
}
.gradio-container { max-width: 1380px !important; margin: 0 auto !important; }

h1, h2, h3, h4 { font-family: var(--serif) !important; font-weight: 500; letter-spacing: -0.01em; color: var(--paper); }

/* ── Masthead: a bound-journal title plate, not a telemetry bar ──────────── */
.cr-header-card {
    position: relative;
    background: var(--ink-raised) !important;
    border: none !important;
    border-bottom: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    padding: 22px 26px 18px !important;
    margin-bottom: 0 !important;
    box-shadow: none !important;
}
/* a single hairline of oxblood sits under the masthead like a binding edge */
.cr-header-card::after {
    content: "";
    position: absolute; left: 0; right: 0; bottom: -1px; height: 2px;
    background: linear-gradient(90deg, var(--oxblood) 0%, var(--oxblood) 18%, transparent 18%);
}
.cr-logo-wrap { display: flex; align-items: baseline; gap: 16px; }
.cr-brand-glyph {
    width: 34px; height: 34px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    color: var(--oxblood-soft);
    border: 1px solid var(--rule);
    border-radius: 2px;
    background: transparent;
    align-self: center;
}
.cr-brand-glyph:hover { border-color: var(--oxblood); color: var(--oxblood-soft); }
.cr-brand-title {
    font-family: var(--serif) !important;
    font-size: 1.7rem !important;
    font-weight: 500 !important;
    color: var(--paper) !important;
    letter-spacing: -0.015em;
    line-height: 1.1;
    display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
}
.cr-title-sub {
    font-family: var(--serif) !important;
    font-style: italic;
    font-weight: 300;
    font-size: 0.98rem;
    color: var(--paper-dim) !important;
    letter-spacing: 0;
}
.cr-header-desc {
    color: var(--paper-dim) !important;
    font-family: var(--serif) !important;
    font-size: 0.95rem;
    font-style: italic;
    margin-top: 7px;
    line-height: 1.45;
    max-width: 62ch;
}
.cr-version-pill {
    font-family: var(--sans);
    font-size: 0.68rem; font-weight: 500;
    padding: 2px 7px; border-radius: 2px;
    background: transparent; border: 1px solid var(--rule);
    color: var(--paper-faint);
}

/* ── Masthead meta (right side): a ledger line, not glowing chips ────────── */
.cr-meta-bar { display: flex; flex-wrap: wrap; gap: 18px; align-items: baseline; justify-content: flex-end; }
.cr-meta-item {
    font-family: var(--sans);
    font-size: 0.8rem; font-weight: 400;
    color: var(--paper-dim);
    background: transparent; border: none; padding: 0;
    display: flex; align-items: baseline; gap: 8px;
}
.cr-meta-item .cr-mono { font-family: var(--serif) !important; font-style: italic; font-size: 0.8rem !important; color: var(--paper-faint); }
.cr-pulse-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--ok); align-self: center;
    box-shadow: 0 0 0 3px rgba(127, 160, 106, 0.14);
}

/* ── Badge pills: flat ink labels with a brass hairline ──────────────────── */
.cr-badge-pill { font-family: var(--sans); font-size: 0.68rem; font-weight: 500; padding: 1px 7px; border-radius: 2px; letter-spacing: 0; }
.cr-badge-orange { background: transparent !important; border: 1px solid var(--oxblood) !important; color: var(--oxblood-soft) !important; }
.cr-badge-mint   { background: transparent !important; border: 1px solid var(--brass-dim) !important; color: var(--brass) !important; }
.cr-badge-subtle { background: transparent !important; border: 1px solid var(--rule-faint) !important; color: var(--paper-faint) !important; }

/* ═══ Pipeline progression — a ruled timeline (a genuine sequence) ═══════════ */
.stepper-container, .cr-stepper-box {
    display: flex; flex-wrap: wrap; gap: 0; align-items: center; justify-content: space-between;
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    padding: 12px 26px !important;
    margin: 0 0 20px 0 !important;
    box-shadow: none !important;
}
.stepper-container > div:first-child { gap: 0 !important; }
.step-node {
    position: relative;
    display: flex; align-items: baseline; gap: 7px;
    padding: 4px 16px 4px 0;
    margin-right: 16px;
    font-family: var(--sans);
    font-size: 0.82rem; font-weight: 400;
    border: none !important;
    color: var(--paper-faint);
}
/* a connecting rule between stations */
.step-node:not(:last-child)::after {
    content: ""; position: absolute; right: 4px; top: 50%; width: 10px; height: 1px;
    background: var(--rule);
}
.step-idx { font-family: var(--serif); font-style: italic; font-size: 0.85rem; color: var(--paper-faint); opacity: 0.8; }
.step-title { font-weight: 400; }
.step-badge { font-size: 0.72rem; color: var(--brass); margin-left: 2px; }
.step-pending { color: var(--paper-faint); }
.step-done    { color: var(--paper-dim); }
.step-done .step-idx { color: var(--brass); opacity: 1; }
.step-active  { color: var(--paper); font-weight: 500; }
.step-active .step-idx { color: var(--oxblood-soft); opacity: 1; }
/* the single orchestrated motion: only the active station marks itself */
.step-active::before {
    content: ""; position: absolute; left: -2px; top: 50%; transform: translateY(-50%);
    width: 3px; height: 14px; background: var(--oxblood);
    animation: alethPulse 1.9s ease-in-out infinite;
}
@keyframes alethPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.cr-stepper-status { font-family: var(--sans); font-size: 0.8rem; color: var(--paper-dim); display: flex; align-items: baseline; gap: 12px; }
.cr-timer-pill { font-family: var(--serif); font-style: italic; font-size: 0.82rem; color: var(--paper-faint); background: transparent; border: none; padding: 0; }

/* ═══ Metrics: a reading ledger, values in serif, hairline-separated ════════ */
.metrics-grid, .cr-scorecard-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
    gap: 0 !important;
    margin: 4px 0 18px 0;
    border: 1px solid var(--rule);
    border-radius: 3px;
    overflow: hidden;
    background: var(--ink-raised);
}
.metric-card, .cr-score-card {
    background: transparent !important;
    border: none !important;
    border-right: 1px solid var(--rule-faint) !important;
    border-radius: 0 !important;
    padding: 13px 15px !important;
    box-shadow: none !important;
    text-align: left;
}
.metric-card:last-child, .cr-score-card:last-child { border-right: none !important; }
.metric-card::after, .cr-score-card::after { display: none !important; }
.metric-card:hover, .cr-score-card:hover { background: rgba(241, 236, 225, 0.02) !important; transform: none !important; box-shadow: none !important; }
.metric-value, .cr-score-val {
    font-family: var(--serif) !important;
    font-size: 1.45rem !important; font-weight: 500;
    color: var(--paper) !important;
    letter-spacing: -0.01em; line-height: 1.1;
    font-variant-numeric: tabular-nums; white-space: nowrap;
}
.metric-label, .cr-score-lbl {
    font-family: var(--sans) !important;
    font-size: 0.68rem !important; font-weight: 500;
    color: var(--paper-faint) !important;
    text-transform: none !important; letter-spacing: 0 !important;
    margin-top: 5px; line-height: 1.3; white-space: normal;
}

/* ═══ Chat canvas — a reading well; synthesis set in serif ══════════════════ */
.gradio-chatbot, #aletheia-chatbot {
    background: var(--ink-sunken) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 4px !important;
    box-shadow: inset 0 1px 0 rgba(0,0,0,0.25) !important;
}
#aletheia-chatbot .message, #aletheia-chatbot .prose, #aletheia-chatbot p, #aletheia-chatbot li {
    font-family: var(--serif) !important;
    font-size: 1.02rem !important;
    line-height: 1.62 !important;
    color: var(--paper) !important;
}
#aletheia-chatbot .prose { max-width: 70ch; }
#aletheia-chatbot h3 { font-family: var(--serif) !important; color: var(--paper); }
#aletheia-chatbot code { font-family: var(--sans) !important; font-size: 0.82rem; background: rgba(183,154,99,0.12); color: var(--brass); padding: 1px 5px; border-radius: 2px; }
#aletheia-chatbot blockquote { border-left: 2px solid var(--oxblood); color: var(--paper-dim); font-style: italic; }

/* ── Primary action — an ink button with an oxblood face, no gradient ────── */
.gr-button-primary, button.primary {
    background: var(--oxblood) !important;
    border: 1px solid var(--oxblood) !important;
    color: #f7efe6 !important;
    font-family: var(--sans) !important;
    font-weight: 500 !important; font-size: 0.9rem !important;
    border-radius: 3px !important;
    box-shadow: none !important;
    transition: background 0.18s ease, transform 0.08s ease !important;
    letter-spacing: 0;
}
.gr-button-primary:hover, button.primary:hover { background: #b2463e !important; transform: none !important; box-shadow: none !important; }
.gr-button-primary:active, button.primary:active { transform: translateY(1px) !important; }

.gr-button-secondary {
    background: transparent !important;
    border: 1px solid var(--rule) !important;
    color: var(--paper-dim) !important;
    font-family: var(--sans) !important; font-size: 0.82rem !important;
    border-radius: 3px !important;
    transition: border-color 0.15s ease, color 0.15s ease !important;
}
.gr-button-secondary:hover { background: transparent !important; color: var(--paper) !important; border-color: var(--brass) !important; }

/* ── Workspace tabs — a reading index, underline marks the place ─────────── */
#workspace-tabs > .tab-nav, #sources-tabs > .tab-nav {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    padding: 0 !important; gap: 0 !important;
    margin-bottom: 14px !important;
}
#workspace-tabs > .tab-nav button, #sources-tabs > .tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 8px 15px !important;
    font-family: var(--sans) !important; font-size: 0.82rem !important; font-weight: 400 !important;
    color: var(--paper-faint) !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
}
#workspace-tabs > .tab-nav button:hover, #sources-tabs > .tab-nav button:hover { color: var(--paper) !important; background: transparent !important; }
#workspace-tabs > .tab-nav button.selected, #sources-tabs > .tab-nav button.selected {
    color: var(--paper) !important;
    border-bottom: 2px solid var(--oxblood) !important;
    background: transparent !important; box-shadow: none !important;
}

/* ── Document catalogue — bibliographic entries, not SaaS cards ──────────── */
.cr-doc-list { display: flex; flex-direction: column; gap: 0; max-height: 230px; overflow-y: auto; margin-top: 8px; padding-right: 4px; }
.cr-doc-card {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--rule-faint) !important;
    border-radius: 0 !important;
    padding: 10px 2px !important;
    display: flex; align-items: center; justify-content: space-between;
    transition: none !important;
}
.cr-doc-card:hover { background: rgba(241, 236, 225, 0.02) !important; transform: none !important; box-shadow: none !important; }
.cr-doc-info { display: flex; flex-direction: column; gap: 4px; width: 100%; min-width: 0 !important; }
.cr-doc-title { font-family: var(--serif) !important; font-size: 0.98rem; font-weight: 400; color: var(--paper); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100% !important; }
.cr-doc-meta { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.cr-doc-empty { font-family: var(--serif); font-style: italic; font-size: 0.92rem; color: var(--paper-faint); padding: 16px; text-align: center; background: transparent; border-radius: 3px; border: 1px dashed var(--rule); }

/* ── Status banners — flat, ink, bordered by meaning ─────────────────────── */
.cr-status-box { padding: 9px 13px; border-radius: 3px; margin: 8px 0; font-family: var(--sans); font-size: 0.85rem; display: flex; align-items: center; gap: 8px; border: 1px solid var(--rule); background: var(--ink-raised); }

.cr-subhead { font-family: var(--sans); font-size: 0.72rem; font-weight: 600; text-transform: none; letter-spacing: 0; color: var(--paper-faint); margin-top: 14px; margin-bottom: 6px; }

/* ── Inputs — paper-well fields with an oxblood focus hairline ───────────── */
textarea, input[type="text"], .gr-box textarea {
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
    font-family: var(--sans) !important;
}
#cr-query-input textarea {
    background: var(--ink-sunken) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
    font-family: var(--serif) !important;
    font-size: 1rem !important;
    color: var(--paper) !important;
    padding: 13px 15px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--oxblood) !important;
    box-shadow: 0 0 0 1px var(--oxblood) !important;
    outline: none !important;
}

.cr-composer-row { align-items: stretch !important; gap: 10px !important; margin-top: 14px !important; }
#cr-submit-btn { align-self: stretch !important; min-height: 100% !important; }
.cr-composer-actions { margin-top: 8px !important; }

/* Examples — a "further reading" index, quiet ruled rows */
#cr-examples .gr-samples-table, #cr-examples table { border: none !important; background: transparent !important; }
#cr-examples button, #cr-examples td {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--rule-faint) !important;
    border-radius: 0 !important;
    color: var(--paper-dim) !important;
    font-family: var(--serif) !important; font-style: italic; font-size: 0.9rem !important;
    transition: color 0.15s ease !important;
    text-align: left !important;
}
#cr-examples button:hover, #cr-examples td:hover { color: var(--oxblood-soft) !important; background: transparent !important; }

#workspace-tabs .tabitem { min-height: 360px !important; }
#metrics-panel { margin-bottom: 4px !important; }
#workspace-tabs .prose, #sources-tabs .prose { font-family: var(--serif) !important; font-size: 0.94rem !important; line-height: 1.6 !important; color: var(--paper-dim); }

/* Evidence provenance list in the source panels */
#cr-graph-preview iframe, #cr-graph-preview > div { border-radius: 3px !important; border: 1px solid var(--rule) !important; }

/* ── Chat welcome / empty state — an invitation set in serif ─────────────── */
.cr-welcome { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 100%; padding: 32px 30px; gap: 8px; }
.cr-welcome-glyph {
    width: 54px; height: 54px; display: flex; align-items: center; justify-content: center;
    border-radius: 3px; color: var(--oxblood-soft);
    background: transparent; border: 1px solid var(--rule);
    margin-bottom: 14px;
}
.cr-welcome-title { font-family: var(--serif); font-size: 1.35rem; font-weight: 500; color: var(--paper); letter-spacing: -0.01em; }
.cr-welcome-sub { font-family: var(--serif); font-size: 0.98rem; font-style: italic; line-height: 1.55; color: var(--paper-dim); max-width: 46ch; margin: 4px auto 0; }
.cr-welcome-hints { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-top: 18px; }
.cr-welcome-pill {
    font-family: var(--sans); font-size: 0.74rem; font-weight: 400;
    color: var(--paper-dim); background: transparent;
    border: 1px solid var(--rule); border-radius: 2px; padding: 5px 11px;
    display: inline-flex; align-items: center; gap: 8px;
}
.cr-pill-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }

/* Scrollbars — thin, brass on hover */
.cr-doc-list::-webkit-scrollbar { width: 7px; }
.cr-doc-list::-webkit-scrollbar-track { background: transparent; }
.cr-doc-list::-webkit-scrollbar-thumb { background: var(--rule); border-radius: 0; }
.cr-doc-list::-webkit-scrollbar-thumb:hover { background: var(--brass); }

#cr-file-uploader { border: 1px dashed var(--rule) !important; border-radius: 3px !important; transition: border-color 0.18s ease !important; }
#cr-file-uploader:hover { border-color: var(--oxblood) !important; background: transparent !important; }

@media (max-width: 1200px) {
    .metrics-grid, .cr-scorecard-grid { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
    .metric-card:nth-child(3), .cr-score-card:nth-child(3) { border-right: none !important; }
}
@media (max-width: 720px) {
    .metrics-grid, .cr-scorecard-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
    .cr-brand-title { font-size: 1.4rem !important; }
    .cr-meta-bar { justify-content: flex-start; margin-top: 10px; }
}

/* Accessibility: honor reduced-motion — kill the one animated mark too */
@media (prefers-reduced-motion: reduce) {
    .step-active::before { animation: none !important; }
    .gr-button-primary, button.primary, .gr-button-secondary { transition: none !important; }
}
"""

# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline State & Formatting Helpers (CodeRabbit Review Style)
# ═══════════════════════════════════════════════════════════════════════════

ALL_STAGES = [
    ("router", "1. Router", "01"),
    ("retriever", "2. Retriever", "02"),
    ("evaluator", "3. Evaluator", "03"),
    ("search_apis", "4. Search APIs", "04"),
    ("pruner", "5. Pruner (AttnRAG)", "05"),
    ("generator", "6. Generator & Critic", "06"),
]

# Rich empty-state shown inside the chat canvas before the first query,
# so the synthesis stream reads as an invitation rather than a black void.
WELCOME_PLACEHOLDER = (
    "<div class='cr-welcome'>"
    "<div class='cr-welcome-glyph'>"
    "<svg width='30' height='30' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
    "stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'>"
    "<circle cx='12' cy='12' r='3'/><path d='M3 12h3M18 12h3M12 3v3M12 18v3'/>"
    "<circle cx='12' cy='12' r='9' stroke-opacity='0.35'/></svg>"
    "</div>"
    "<div class='cr-welcome-title'>Begin a reading</div>"
    "<div class='cr-welcome-sub'>Pose a question about the papers in your library. "
    "The answer is written here as it comes together — retrieved, weighed, "
    "compressed, and checked against its sources.</div>"
    "<div class='cr-welcome-hints'>"
    "<span class='cr-welcome-pill'><span class='cr-pill-dot' style='background:#a23c35'></span>Searches three ways at once</span>"
    "<span class='cr-welcome-pill'><span class='cr-pill-dot' style='background:#b79a63'></span>Scores the evidence</span>"
    "<span class='cr-welcome-pill'><span class='cr-pill-dot' style='background:#7fa06a'></span>Checks every claim</span>"
    "</div>"
    "</div>"
)


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
    """Render precision automated review checkpoints timeline."""
    completed = set(completed_nodes or [])

    nodes_html = []
    for node_id, label, icon in ALL_STAGES:
        if node_id == active_node:
            state_cls = "step-active"
            badge = "active"
        elif node_id in completed:
            state_cls = "step-done"
            badge = "✓"
        else:
            state_cls = "step-pending"
            badge = ""

        badge_span = (
            f"<span class='step-badge'>{badge}</span>"
            if badge
            else ""
        )
        nodes_html.append(
            f"<div class='step-node {state_cls}'>"
            f"<span class='step-idx'>{icon}</span> <span class='step-title'>{label}</span> {badge_span}"
            f"</div>"
        )

    timer_text = f"{elapsed_time:.1f}s" if elapsed_time > 0 else "0.0s"
    status_badge = (
        f"<div class='cr-stepper-status'>"
        f"<span>Status: {html.escape(status_label)}</span>"
        f"<span class='cr-timer-pill'>{timer_text}</span>"
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
    """Render scientific review scorecard cards: Rubric Score, AttentionRAG Compression, Latency, Critic."""
    s = _ensure_state(state)
    if not s:
        return (
            "<div class='metrics-grid cr-scorecard-grid'>"
            "<div class='metric-card'><div class='metric-value' style='color:#8d8676;'>—</div><div class='metric-label'>Evidence score</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#8d8676;'>—</div><div class='metric-label'>Context compression</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#8d8676;'>0.0s</div><div class='metric-label'>Time to answer</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#8d8676;'>—</div><div class='metric-label'>Fact check</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#8d8676;'>$0.00</div><div class='metric-label'>Token cost</div></div>"
            "</div>"
        )

    # 1. Evidence Score
    ev_score = s.evaluation.rubric_score if s.evaluation else 0
    ev_color = "#7fa06a" if ev_score >= 80 else ("#c9972f" if ev_score >= 50 else "#b6443c")
    ev_text = f"{ev_score}/100"

    # 2. Compression Ratio
    comp_ratio = s.compression.compression_ratio if s.compression else 0.0
    comp_text = f"{comp_ratio:.1f}x" if comp_ratio > 0 else "1.0x"

    # 3. Latency
    latency_text = f"{elapsed:.2f}s"

    # 4. Critic Verdict
    feedback = s.generation.critic_feedback if s.generation else None
    if not feedback or "approved" in feedback.lower():
        critic_icon = "Approved"
        critic_color = "#7fa06a"
    elif "regenerate" in feedback.lower() or "retry" in feedback.lower():
        critic_icon = "Regenerated"
        critic_color = "#c9972f"
    else:
        critic_icon = "Passed"
        critic_color = "#7fa06a"

    # 5. Cost
    cost_text = f"${s.pipeline_cost:.4f}"

    return (
        f"<div class='metrics-grid cr-scorecard-grid'>"
        f"<div class='metric-card'><div class='metric-value' style='color:{ev_color};'>{ev_text}</div><div class='metric-label'>Evidence score</div></div>"
        f"<div class='metric-card'><div class='metric-value'>{comp_text}</div><div class='metric-label'>Context compression</div></div>"
        f"<div class='metric-card'><div class='metric-value'>{latency_text}</div><div class='metric-label'>Time to answer</div></div>"
        f"<div class='metric-card'><div class='metric-value' style='font-size:1.15rem; color:{critic_color};'>{critic_icon}</div><div class='metric-label'>Fact check</div></div>"
        f"<div class='metric-card'><div class='metric-value'>{cost_text}</div><div class='metric-label'>Token cost</div></div>"
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

    items = [
        canvas_html,
        "\n<div style='margin-top:14px; font-size:0.80rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; color:var(--cr-orange);'>Active Knowledge Graph Triples</div>"
    ]
    for i, triple in enumerate(triples[:15], 1):
        # Triples derive from ingested document content (untrusted) — escape
        # before placing into markdown that Gradio renders as HTML (F4).
        items.append(f"- `triple:{i:02d}` **{html.escape(str(triple))}**")
    return "\n".join(items)


def format_faiss_markdown(state: Any) -> str:
    """Format FAISS dense vector search results with CodeRabbit review pill aesthetics."""
    s = _ensure_state(state)
    if not s or not s.memory or not s.memory.vector_chunks:
        return "*(No FAISS vector chunks retrieved yet.)*"

    items = []
    for i, chunk in enumerate(s.memory.vector_chunks[:5], 1):
        preview = chunk[:400].strip() + ("..." if len(chunk) > 400 else "")
        # Retrieved chunk text is untrusted (from ingested PDFs) — escape it (F4).
        items.append(
            f"**Chunk #{i}:** `FAISS Dense Vector`  \n"
            f"> {html.escape(preview)}\n"
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
        # Retrieved chunk text is untrusted (from ingested PDFs) — escape it (F4).
        items.append(
            f"**Lexical Match #{i}:** `BM25Okapi Sparse`  \n"
            f"> {html.escape(preview)}\n"
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
        # External API content is untrusted — escape before markdown render (F4).
        items.append(f"**External Literature #{i}:**\n> {html.escape(preview)}\n")
    return "\n\n".join(items)


def render_evidence_breakdown_html(
    evidence_breakdown: Dict[str, int] | None = None,
    critic_confidence: float | None = None,
    source_map: Dict[str, str] | None = None,
) -> str:
    """Render evidence transparency (Feature 7.1): the 100-pt rubric sub-scores as
    labeled bars, the critic confidence, and the click-through source map.

    Theme-safe: uses currentColor / inherited colors so it reads in light or dark.
    """
    evidence_breakdown = evidence_breakdown or {}
    source_map = source_map or {}

    maxima = {
        "retrieval_confidence": 40,
        "answer_specificity": 25,
        "source_diversity": 15,
        "metadata_completeness": 10,
        "recency": 10,
    }
    labels = {
        "retrieval_confidence": "Retrieval Confidence",
        "answer_specificity": "Answer Specificity",
        "source_diversity": "Source Diversity",
        "metadata_completeness": "Metadata Completeness",
        "recency": "Recency",
    }

    rows = []
    for key, mx in maxima.items():
        val = int(evidence_breakdown.get(key, 0))
        pct = int(100 * val / mx) if mx else 0
        rows.append(
            f'<div style="margin:4px 0;font-size:12px">'
            f'<div style="display:flex;justify-content:space-between">'
            f'<span>{labels[key]}</span><span>{val}/{mx}</span></div>'
            f'<div style="background:rgba(127,127,127,0.25);border-radius:4px;height:7px">'
            f'<div style="width:{pct}%;height:7px;border-radius:4px;background:currentColor;opacity:0.7"></div>'
            f'</div></div>'
        )

    conf_html = ""
    if critic_confidence is not None:
        conf_pct = int(100 * float(critic_confidence))
        conf_html = (
            f'<div style="margin-top:8px;font-size:12px">'
            f'Critic confidence: <strong>{conf_pct}%</strong></div>'
        )

    sources_html = ""
    if source_map:
        items = "".join(
            # tag + source text are untrusted (derived from retrieved content) —
            # escape both before interpolating into raw HTML (F4).
            f'<li style="margin:2px 0"><code>[{html.escape(str(tag))}]</code> '
            f'{html.escape(text[:140] + ("…" if len(text) > 140 else ""))}</li>'
            for tag, text in list(source_map.items())[:12]
        )
        sources_html = (
            f'<details style="margin-top:8px;font-size:12px">'
            f'<summary>Sources ({len(source_map)})</summary>'
            f'<ul style="margin:4px 0 0 0;padding-left:18px">{items}</ul></details>'
        )

    return (
        '<div style="border:1px solid rgba(127,127,127,0.3);border-radius:8px;padding:10px">'
        '<div style="font-weight:600;font-size:13px;margin-bottom:6px">Evidence Transparency</div>'
        + "".join(rows) + conf_html + sources_html + "</div>"
    )


def render_header_meta_html(catalog: Dict[str, Any] | None = None) -> str:
    """Render precision telemetry bar with real-time ingested paper counts."""
    if catalog is None:
        try:
            catalog = ingestion_service.get_catalog()
        except Exception:
            catalog = {"total_documents": 0, "total_chunks": 0}
    doc_count = catalog.get("total_documents", 0)
    chunk_count = catalog.get("total_chunks", 0)
    papers_word = "paper" if doc_count == 1 else "papers"
    return (
        f"<div class='cr-meta-bar'>"
        f"<div class='cr-meta-item'><span class='cr-pulse-dot'></span> "
        f"<span>{doc_count} {papers_word} in the library</span></div>"
        f"<div class='cr-meta-item'><span class='cr-mono'>{chunk_count} indexed passages</span></div>"
        f"<div class='cr-version-pill'>v1.0.0</div>"
        f"</div>"
    )


def render_ingest_status_html(res: Dict[str, Any] | None = None) -> str:
    """Render status pill for in-browser PDF ingestion and memory wipe operations."""
    if res is None:
        return (
            "<div class='cr-status-box'>"
            "<span style='color:#8d8676;'>Drop a PDF here, or choose a file, to add it to the library.</span>"
            "</div>"
        )

    status = res.get("status", "")
    msg = html.escape(res.get("message", ""))
    neo4j_st = res.get("neo4j_status", "")

    if status == "success":
        neo4j_badge = ""
        if neo4j_st == "offline":
            neo4j_badge = (
                "&nbsp;<span class='cr-badge-pill cr-badge-subtle'>"
                "Graph layer offline — dense &amp; lexical search active</span>"
            )
        elif neo4j_st == "synced":
            neo4j_badge = (
                "&nbsp;<span class='cr-badge-pill cr-badge-mint'>Graph synced</span>"
            )
        return (
            f"<div class='cr-status-box' style='border-left:3px solid #7fa06a; color:#aebd9e;'>"
            f"<span>{msg}</span>{neo4j_badge}"
            f"</div>"
        )
    elif status == "wiped":
        return (
            f"<div class='cr-status-box' style='border-left:3px solid #c9972f; color:#d6b878;'>"
            f"<span>{msg}</span>"
            f"</div>"
        )
    elif status in ("skipped", "no_files"):
        return (
            f"<div class='cr-status-box' style='border-left:3px solid #b79a63; color:#cbc3b4;'>"
            f"<span>{msg}</span>"
            f"</div>"
        )
    else:
        return (
            f"<div class='cr-status-box' style='border-left:3px solid #a23c35; color:#c65a4f;'>"
            f"<span>{msg}</span>"
            f"</div>"
        )


def render_catalog_html(catalog: Dict[str, Any] | None = None) -> str:
    """Render structured catalog list of ingested documents."""
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
            f"<div class='cr-doc-title' title='{title}'>{title}</div>"
            f"<div class='cr-doc-meta'>"
            f"<span class='cr-badge-pill cr-badge-orange'>{chunks} Chunks</span>"
            f"<span class='cr-badge-pill cr-badge-mint'>{pages} Pages</span>"
            f"<span class='cr-badge-pill cr-badge-subtle'>{ts}</span>"
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
    """Wrap synthesized answer into structured scientific review format."""
    status_icon = "Verified" if evidence_score >= 80 else "Caution"
    rubric_badge = f"`Evidence Rubric: {evidence_score}/100`"
    attn_badge = f"`AttentionRAG: {compression_ratio:.1f}x`"
    critic_badge = f"`Critic: {critic_verdict}`"

    header_block = (
        f"### Research Synthesis & Verification\n\n"
        f"> **Assessment:** **{status_icon}** ({critic_verdict}) &nbsp;&bull;&nbsp; {rubric_badge} &nbsp;&bull;&nbsp; {attn_badge} &nbsp;&bull;&nbsp; `Latency: {latency:.1f}s`\n\n"
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

    # Initialize conversation message in precision scientific format
    user_msg = {"role": "user", "content": clean_query}
    assistant_msg = {
        "role": "assistant",
        "content": "*Synthesizing evidence across Three-Layer Global Memory...*",
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

    # Stage descriptions for real-time trace
    stage_messages = {
        "router": "*Classifying query intent and canonical domain terms...*",
        "retriever": "*Retrieving passages from Three-Layer Global Memory (Neo4j, FAISS, BM25)...*",
        "evaluator": "*Evaluating 100-point evidence sufficiency rubric...*",
        "search_apis": "*Querying external literature fallback (Crossref, Semantic Scholar, arXiv)...*",
        "pruner": "*Compressing context micro-windows via AttentionRAG...*",
        "generator": "*Drafting cited synthesis and executing The Critic QA audit...*",
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

        # Final completion yield formatted with scientific review summary
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
                f"Completed in {total_time:.1f}s",
            ),
            format_neo4j_markdown(latest_state),
            format_faiss_markdown(latest_state),
            format_bm25_markdown(latest_state),
            format_external_markdown(latest_state),
            render_metrics_html(latest_state, total_time),
        )

    except Exception as exc:
        logger.error("Pipeline streaming failure: %s", exc, exc_info=True)
        error_msg = f"**Review Pipeline Interrupted:** `{html.escape(str(exc))}`\n\n*Graceful degradation active: Inspect system logs for diagnostics.*"
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
#  Gradio UI Layout Construction (Precision Research Console)
# ═══════════════════════════════════════════════════════════════════════════

def _build_theme() -> "gr.themes.Base":
    """Reading-room base theme so Gradio's own components (inputs, buttons,
    blocks) render on warm ink surfaces at the source — the custom CSS then
    layers the bound-journal aesthetic (serif reading type, oxblood accent,
    brass rules) on top."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.red,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.stone,
        font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
        font_mono=(gr.themes.GoogleFont("Spectral"), "Georgia", "serif"),
    ).set(
        body_background_fill="#15191c",
        body_background_fill_dark="#15191c",
        body_text_color="#f1ece1",
        background_fill_primary="#1c2227",
        background_fill_secondary="#11161a",
        block_background_fill="#1c2227",
        block_border_color="rgba(183, 154, 99, 0.20)",
        block_label_background_fill="#1c2227",
        block_label_text_color="#8d8676",
        border_color_primary="rgba(183, 154, 99, 0.20)",
        input_background_fill="#11161a",
        input_background_fill_focus="#11161a",
        input_border_color="rgba(183, 154, 99, 0.20)",
        input_border_color_focus="#a23c35",
        input_placeholder_color="#8d8676",
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_hover="transparent",
        button_secondary_text_color="#cbc3b4",
        button_secondary_border_color="rgba(183, 154, 99, 0.20)",
        color_accent_soft="rgba(162, 60, 53, 0.14)",
    )
    return theme


def create_ui() -> gr.Blocks:
    """Construct the precision research console web app for Project Aletheia."""

    with gr.Blocks(
        title="Aletheia — a reading room for AI research",
        theme=_build_theme(),
    ) as demo:
        # Injected Custom Design System CSS
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # ── Header: Research Console Telemetry & Brandmark ────────────────────
        with gr.Row(elem_classes=["cr-header-card"]):
            with gr.Column(scale=8):
                gr.HTML(
                    """
                    <div class="cr-logo-wrap">
                        <div class="cr-brand-glyph">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="3"/>
                                <path d="M3 12h3M18 12h3M12 3v3M12 18v3"/>
                                <circle cx="12" cy="12" r="9" stroke-opacity="0.35"/>
                            </svg>
                        </div>
                        <div>
                            <div class="cr-brand-title">
                                Aletheia <span class="cr-title-sub">a reading room for AI research</span>
                            </div>
                            <div class="cr-header-desc">
                                Ask a question about your library of papers. Aletheia retrieves the
                                evidence, compresses it, checks every claim against its source, and
                                writes back a cited answer.
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

        # ── Checkpoints Stepper ──────────────────────────────────────────────
        status_stepper = gr.HTML(
            value=render_stepper_html(status_label="System Ready"),
            elem_id="pipeline-stepper",
        )

        # ── Main Research Workspace (2-Column Review Layout) ─────────────────
        with gr.Row():
            # Left Column: Synthesis Stream & Query Input
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="Synthesis Stream & Verification Trace",
                    height=460,
                    elem_id="aletheia-chatbot",
                    layout="bubble",
                    show_label=False,
                    placeholder=WELCOME_PLACEHOLDER,
                )

                with gr.Row(elem_classes=["cr-composer-row"]):
                    query_input = gr.Textbox(
                        placeholder="Ask the research console  —  e.g. How does AttentionRAG compress context while preserving citation fidelity?",
                        lines=2,
                        max_lines=6,
                        scale=9,
                        label="Literature Query",
                        show_label=False,
                        elem_id="cr-query-input",
                        autofocus=True,
                    )
                    submit_btn = gr.Button("Run Synthesis", variant="primary", scale=3, elem_id="cr-submit-btn")

                with gr.Row(elem_classes=["cr-composer-actions"]):
                    clear_btn = gr.Button("Reset Conversation", variant="secondary", size="sm")

                # Verified Research Query Presets
                gr.Examples(
                    examples=[
                        "How does AttentionRAG compress context while preserving citation fidelity?",
                        "Compare MemGraphRAG hierarchical indexing with TechGraphRAG multi-agent retrieval.",
                        "What are the core components of the Three-Layer Global Memory architecture?",
                        "How does The Critic audit draft answers against passage provenance?",
                    ],
                    inputs=query_input,
                    label="Try a research query",
                    elem_id="cr-examples",
                )

            # Right Column: Review Scorecard & Multi-Source Breakdown
            with gr.Column(scale=5):
                metrics_display = gr.HTML(
                    value=render_metrics_html(None),
                    elem_id="metrics-panel",
                )

                # ── Unified Workspace Tabs ───────────────────────────────────
                with gr.Tabs(elem_id="workspace-tabs"):

                    # Tab 1: Memory & Knowledge Graph
                    with gr.Tab("Memory & Graph", id="tab-memory"):
                        with gr.Tabs(elem_id="sources-tabs"):
                            with gr.Tab("Neo4j Ontology"):
                                sources_neo4j = gr.Markdown(
                                    value=format_neo4j_markdown(None),
                                    label="Neo4j Knowledge Graph Triples",
                                    elem_id="cr-graph-preview",
                                )
                            with gr.Tab("FAISS Dense"):
                                sources_faiss = gr.Markdown(
                                    value="*(Run a query to inspect top cosine similarity chunks)*",
                                    label="FAISS Dense Vectors",
                                )
                            with gr.Tab("BM25 Lexical"):
                                sources_bm25 = gr.Markdown(
                                    value="*(Run a query to inspect BM25Okapi lexical matches)*",
                                    label="BM25 Sparse Lexical Store",
                                )
                            with gr.Tab("External APIs"):
                                sources_external = gr.Markdown(
                                    value="*(External search triggers when internal evidence score < 80)*",
                                    label="External Academic APIs",
                                )

                    # Tab 2: Live Ingestion & Autonomous ArXiv Watcher
                    with gr.Tab("Ingest & Watcher", id="tab-ingest"):
                        file_uploader = gr.File(
                            label="Select or Drop Academic Research Papers (PDF)",
                            file_types=[".pdf"],
                            file_count="multiple",
                            elem_id="cr-file-uploader",
                        )
                        with gr.Row():
                            ingest_btn = gr.Button("Ingest Papers into Memory", variant="primary", scale=8)
                            wipe_btn = gr.Button("Wipe All Memory", variant="stop", scale=4)

                        # ── Autonomous ArXiv Watcher Controls ──────────────────
                        with gr.Accordion("Autonomous ArXiv Daily Watcher", open=False):
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
                            arxiv_btn = gr.Button("Fetch & Ingest Trending ArXiv Papers", variant="secondary")

                        ingest_status = gr.HTML(
                            value=render_ingest_status_html(None),
                            elem_id="cr-ingest-status",
                        )

                        gr.HTML("<div class='cr-subhead'>Active Literature in Memory:</div>")
                        catalog_display = gr.HTML(
                            value=render_catalog_html(),
                            elem_id="cr-catalog-display",
                        )

                    # Tab 3: Settings Drawer
                    with gr.Tab("Settings", id="tab-settings"):
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
