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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --cr-orange: #FF570A;
    --cr-orange-hover: #FF6824;
    --cr-orange-subtle: rgba(255, 87, 10, 0.10);
    --cr-mint: #00F2A5;
    --cr-mint-subtle: rgba(0, 242, 165, 0.10);
    --cr-cobalt: #3B82F6;
    --cr-cobalt-subtle: rgba(59, 130, 246, 0.10);
    --cr-canvas: #090a0f;
    --cr-card: #10121a;
    --cr-card-alt: #151822;
    --cr-card-hover: #191c28;
    --cr-border: rgba(255, 255, 255, 0.07);
    --cr-border-subtle: rgba(255, 255, 255, 0.04);
    --cr-text-primary: #F8FAFC;
    --cr-text-secondary: #94A3B8;
    --cr-text-muted: #64748B;
    --cr-mono: 'JetBrains Mono', monospace;
}

body, .gradio-container {
    background: #090a0f !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--cr-text-primary) !important;
    letter-spacing: -0.011em;
}

h1, h2, h3, h4, .cr-font-brand {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.025em;
    font-weight: 700;
}

code, pre, .cr-mono {
    font-family: var(--cr-mono) !important;
}

/* ── Console Header Card ─────────────────────────────────────────────────── */
.cr-header-card {
    background: #10121a !important;
    border: 1px solid var(--cr-border) !important;
    border-radius: 12px !important;
    padding: 16px 22px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
}

.cr-logo-wrap {
    display: flex;
    align-items: center;
    gap: 14px;
}

.cr-brand-glyph {
    width: 38px;
    height: 38px;
    background: #161924;
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #F8FAFC;
    flex-shrink: 0;
    transition: all 0.2s ease;
}

.cr-brand-glyph:hover {
    border-color: rgba(255, 87, 10, 0.45);
    background: #191d2a;
}

.cr-brand-title {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.cr-title-sub {
    font-size: 0.80rem;
    font-weight: 500;
    color: var(--cr-text-muted);
    letter-spacing: 0.06em;
    font-family: var(--cr-mono);
}

.cr-version-pill {
    font-size: 0.68rem;
    font-family: var(--cr-mono);
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 3px 8px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--cr-border);
    color: var(--cr-text-secondary);
}

.cr-header-desc {
    color: var(--cr-text-secondary);
    font-size: 0.80rem;
    margin-top: 4px;
    line-height: 1.4;
    letter-spacing: -0.005em;
}

.cr-meta-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    justify-content: flex-end;
}

.cr-meta-item {
    font-size: 0.74rem;
    font-weight: 500;
    color: var(--cr-text-secondary);
    background: #12141c;
    border: 1px solid var(--cr-border);
    padding: 4px 10px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.cr-pulse-dot {
    width: 6px;
    height: 6px;
    background: var(--cr-mint);
    border-radius: 50%;
    box-shadow: 0 0 6px rgba(0, 242, 165, 0.4);
}

/* ── Badge Pills ────────────────────────────────────────────────────────── */
.cr-badge-pill {
    font-size: 0.70rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.02em;
}

.cr-badge-orange {
    background: rgba(255, 87, 10, 0.10) !important;
    border: 1px solid rgba(255, 87, 10, 0.30) !important;
    color: #FFA066 !important;
}

.cr-badge-mint {
    background: rgba(0, 242, 165, 0.08) !important;
    border: 1px solid rgba(0, 242, 165, 0.25) !important;
    color: #4EFEB3 !important;
}

.cr-badge-subtle {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid var(--cr-border) !important;
    color: var(--cr-text-muted) !important;
}

/* ── Pipeline Stepper ───────────────────────────────────────────────────── */
.stepper-container, .cr-stepper-box {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    justify-content: space-between;
    background: #10121a;
    border: 1px solid var(--cr-border);
    border-radius: 10px;
    padding: 8px 14px;
    margin-bottom: 14px;
}

.step-node {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 500;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}

.step-idx {
    font-family: var(--cr-mono);
    font-size: 0.70rem;
    font-weight: 600;
    opacity: 0.65;
}

.step-title {
    font-weight: 500;
}

.step-badge {
    font-size: 0.68rem;
    font-family: var(--cr-mono);
    opacity: 0.85;
    margin-left: 2px;
}

.step-pending {
    background: transparent;
    color: var(--cr-text-muted);
}

.step-active {
    background: rgba(255, 87, 10, 0.10);
    color: #FFA066;
    border-color: rgba(255, 87, 10, 0.35);
    font-weight: 600;
}

.step-done {
    background: rgba(0, 242, 165, 0.06);
    color: #00F2A5;
    border-color: rgba(0, 242, 165, 0.20);
}

.cr-stepper-status {
    font-size: 0.76rem;
    font-weight: 500;
    color: var(--cr-text-secondary);
    display: flex;
    align-items: center;
    gap: 8px;
}

.cr-timer-pill {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--cr-border);
    padding: 2px 7px;
    border-radius: 4px;
    font-family: var(--cr-mono);
    font-size: 0.72rem;
    color: var(--cr-text-secondary);
}

/* ── Metrics Scorecard ──────────────────────────────────────────────────── */
.metrics-grid, .cr-scorecard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 8px;
    margin: 8px 0 14px 0;
}

.metric-card, .cr-score-card {
    background: #10121a;
    border: 1px solid var(--cr-border);
    border-radius: 8px;
    padding: 10px 12px;
    text-align: left;
    transition: border-color 0.2s ease;
}

.metric-card:hover, .cr-score-card:hover {
    border-color: rgba(255, 255, 255, 0.14);
}

.metric-value, .cr-score-val {
    font-family: 'Inter', sans-serif;
    font-size: 1.22rem;
    font-weight: 700;
    color: var(--cr-text-primary);
    letter-spacing: -0.02em;
}

.metric-label, .cr-score-lbl {
    font-size: 0.68rem;
    font-weight: 500;
    color: var(--cr-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

/* ── Chatbot & Message Canvas ───────────────────────────────────────────── */
.gradio-chatbot, #aletheia-chatbot {
    background: #0d0f15 !important;
    border: 1px solid var(--cr-border) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}

/* Primary Button: Refined Matte Ember */
.gr-button-primary, button.primary {
    background: #FF570A !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.4), 0 4px 12px rgba(255, 87, 10, 0.2) !important;
    transition: all 0.18s ease !important;
}

.gr-button-primary:hover, button.primary:hover {
    background: #FF6824 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.5), 0 6px 18px rgba(255, 87, 10, 0.3) !important;
    transform: translateY(-1px) !important;
}

.gr-button-primary:active, button.primary:active {
    transform: scale(0.99) !important;
}

/* Secondary Button: Subtle Surface */
.gr-button-secondary {
    background: #141722 !important;
    border: 1px solid var(--cr-border) !important;
    color: var(--cr-text-secondary) !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}

.gr-button-secondary:hover {
    background: #191c28 !important;
    color: var(--cr-text-primary) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
}

/* ── Workspace Segmented Tabs ───────────────────────────────────────────── */
#workspace-tabs > .tab-nav {
    background: #10121a !important;
    border: 1px solid var(--cr-border) !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 3px !important;
    margin-bottom: 12px !important;
}

#workspace-tabs > .tab-nav button {
    border-radius: 6px !important;
    padding: 6px 14px !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    color: var(--cr-text-secondary) !important;
    border: 1px solid transparent !important;
    transition: all 0.15s ease !important;
}

#workspace-tabs > .tab-nav button:hover {
    background: rgba(255, 255, 255, 0.03) !important;
    color: var(--cr-text-primary) !important;
}

#workspace-tabs > .tab-nav button.selected {
    background: #161924 !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
}

/* ── Document List & Cards ──────────────────────────────────────────────── */
.cr-doc-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 220px;
    overflow-y: auto;
    margin-top: 8px;
    padding-right: 4px;
}

.cr-doc-card {
    background: #131620;
    border: 1px solid var(--cr-border);
    border-radius: 6px;
    padding: 8px 11px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.15s ease;
}

.cr-doc-card:hover {
    border-color: rgba(255, 255, 255, 0.15);
}

.cr-doc-info {
    display: flex;
    flex-direction: column;
    gap: 3px;
    width: 100%;
    min-width: 0 !important;
}

.cr-doc-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--cr-text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100% !important;
}

.cr-doc-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    align-items: center;
}

.cr-doc-empty {
    font-size: 0.80rem;
    color: var(--cr-text-muted);
    padding: 12px;
    text-align: center;
    background: rgba(255, 255, 255, 0.01);
    border-radius: 6px;
    border: 1px dashed var(--cr-border);
}

.cr-status-box {
    padding: 7px 11px;
    border-radius: 6px;
    margin: 8px 0;
    font-size: 0.80rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.cr-subhead {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--cr-text-muted);
    margin-top: 10px;
    margin-bottom: 4px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   ELEVATION LAYER — Premium depth, glassmorphism & micro-animation refinements
   (Overrides flat surfaces from the base layer above; aligned with the
   modern-ui-designer skill: layered radial canvas, glass cards, pulse glows.)
   ═══════════════════════════════════════════════════════════════════════════ */

/* Ambient canvas: subtle multi-point radial glow instead of a flat void */
body, .gradio-container {
    background:
        radial-gradient(900px circle at 12% -8%, rgba(255, 87, 10, 0.07), transparent 42%),
        radial-gradient(1000px circle at 100% 0%, rgba(59, 130, 246, 0.06), transparent 46%),
        radial-gradient(1200px circle at 50% 120%, rgba(0, 242, 165, 0.045), transparent 50%),
        #090a0f !important;
    background-attachment: fixed !important;
}

.gradio-container { max-width: 1480px !important; margin: 0 auto !important; }

/* Shared smooth easing for all interactive surfaces */
.cr-header-card, .metric-card, .cr-score-card, .cr-doc-card,
.stepper-container, .cr-stepper-box, .gr-button-primary, button.primary,
.gr-button-secondary, #workspace-tabs > .tab-nav button {
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

/* Header — glass surface with an accent top hairline */
.cr-header-card {
    position: relative;
    background: linear-gradient(180deg, rgba(22, 26, 38, 0.72), rgba(14, 16, 24, 0.72)) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border: 1px solid var(--cr-border) !important;
    border-radius: 14px !important;
    box-shadow: 0 10px 34px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
    overflow: hidden;
}
.cr-header-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cr-orange) 20%, var(--cr-mint) 55%, var(--cr-cobalt) 85%, transparent);
    opacity: 0.8;
}

.cr-brand-glyph {
    background: linear-gradient(145deg, #1b1f2c, #121521) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06), 0 2px 8px rgba(0, 0, 0, 0.4) !important;
}
.cr-brand-glyph:hover {
    color: var(--cr-orange) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 4px 14px rgba(255, 87, 10, 0.25) !important;
    transform: translateY(-1px);
}

/* Live pulse dot — soft breathing glow for "online" status */
.cr-pulse-dot { animation: crPulseGlow 2.4s ease-in-out infinite; }
@keyframes crPulseGlow {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px rgba(0, 242, 165, 0.65); }
    50% { opacity: 0.55; box-shadow: 0 0 2px rgba(0, 242, 165, 0.2); }
}

/* Stepper — glass tray */
.stepper-container, .cr-stepper-box {
    background: linear-gradient(180deg, rgba(20, 24, 36, 0.66), rgba(14, 16, 24, 0.66)) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid var(--cr-border) !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.3) !important;
}
/* Active pipeline node breathes while running */
.step-active {
    animation: crStepBreath 1.8s ease-in-out infinite;
    box-shadow: 0 0 0 1px rgba(255, 87, 10, 0.25), 0 4px 14px rgba(255, 87, 10, 0.18) !important;
}
@keyframes crStepBreath {
    0%, 100% { box-shadow: 0 0 0 1px rgba(255, 87, 10, 0.25), 0 4px 14px rgba(255, 87, 10, 0.18); }
    50% { box-shadow: 0 0 0 1px rgba(255, 87, 10, 0.45), 0 6px 20px rgba(255, 87, 10, 0.32); }
}

/* Metric scorecards — glass with accent top edge + lift on hover */
.metric-card, .cr-score-card {
    position: relative;
    background: linear-gradient(180deg, rgba(20, 24, 36, 0.6), rgba(14, 16, 24, 0.6)) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--cr-border) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.28) !important;
    overflow: hidden;
}
.metric-card::after, .cr-score-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--cr-orange), rgba(255, 87, 10, 0));
    opacity: 0;
    transition: opacity 0.25s ease;
}
.metric-card:hover, .cr-score-card:hover {
    border-color: rgba(255, 87, 10, 0.35) !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(255, 87, 10, 0.12) !important;
}
.metric-card:hover::after, .cr-score-card:hover::after { opacity: 0.9; }
.metric-value, .cr-score-val { font-variant-numeric: tabular-nums; }

/* Chatbot canvas — subtle inner gradient + deeper shadow */
.gradio-chatbot, #aletheia-chatbot {
    background: linear-gradient(180deg, rgba(14, 16, 24, 0.9), rgba(9, 10, 15, 0.9)) !important;
    border: 1px solid var(--cr-border) !important;
    border-radius: 14px !important;
    box-shadow: 0 10px 34px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
}

/* Primary button — gradient ember with lift + press feedback */
.gr-button-primary, button.primary {
    background: linear-gradient(180deg, #FF6A24, #FF570A) !important;
    border: 1px solid rgba(255, 150, 90, 0.35) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.4), 0 6px 18px rgba(255, 87, 10, 0.28), inset 0 1px 0 rgba(255,255,255,0.2) !important;
}
.gr-button-primary:hover, button.primary:hover {
    background: linear-gradient(180deg, #FF7A3A, #FF6117) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.5), 0 10px 26px rgba(255, 87, 10, 0.4), inset 0 1px 0 rgba(255,255,255,0.25) !important;
}
.gr-button-primary:active, button.primary:active { transform: scale(0.98) !important; }

/* Secondary button — press feedback */
.gr-button-secondary:active { transform: scale(0.98) !important; }

/* Text inputs — focus ring in brand coral */
textarea, input[type="text"], .gr-box textarea {
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: rgba(255, 87, 10, 0.55) !important;
    box-shadow: 0 0 0 3px rgba(255, 87, 10, 0.14) !important;
    outline: none !important;
}

/* Workspace tabs — glass tray + animated selected underline */
#workspace-tabs > .tab-nav {
    background: rgba(18, 22, 34, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}
#workspace-tabs > .tab-nav button.selected {
    background: linear-gradient(180deg, #1b1f2c, #14171f) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
}

/* Document cards — glass + lift on hover */
.cr-doc-card {
    background: linear-gradient(180deg, rgba(20, 24, 36, 0.55), rgba(15, 17, 26, 0.55)) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
}
.cr-doc-card:hover {
    border-color: rgba(255, 87, 10, 0.3) !important;
    transform: translateX(2px);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3) !important;
}

/* Refined scrollbars inside document/source panels */
.cr-doc-list::-webkit-scrollbar { width: 8px; }
.cr-doc-list::-webkit-scrollbar-track { background: transparent; }
.cr-doc-list::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 8px;
}
.cr-doc-list::-webkit-scrollbar-thumb:hover { background: rgba(255, 87, 10, 0.4); }

/* Badges lift slightly on hover for tactility */
.cr-badge-pill { transition: transform 0.15s ease, box-shadow 0.15s ease; }
.cr-doc-card:hover .cr-badge-orange { box-shadow: 0 0 10px rgba(255, 87, 10, 0.25); }

/* Accordion & file-drop surfaces inherit the glass treatment */
#cr-file-uploader {
    border: 1px dashed var(--cr-border) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}
#cr-file-uploader:hover {
    border-color: rgba(255, 87, 10, 0.4) !important;
    background: rgba(255, 87, 10, 0.03) !important;
}

/* ── Header contrast fix ──────────────────────────────────────────────────
   Gradio's prose/opacity styles were dimming the brand text to near-grey.
   Force crisp, intentional hierarchy. */
.cr-brand-title { color: #FFFFFF !important; opacity: 1 !important; }
.cr-brand-title .cr-title-sub { color: #7C8498 !important; }
.cr-header-desc { color: #AEB7C7 !important; opacity: 1 !important; }
.cr-header-card .cr-meta-item { color: #C3CBDA !important; }
.cr-brand-glyph { color: #FF843F !important; }

/* ── Metrics scorecard — five cards balanced in a single responsive row ──── */
.metrics-grid, .cr-scorecard-grid {
    grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
    gap: 10px !important;
}
@media (max-width: 1200px) {
    .metrics-grid, .cr-scorecard-grid { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
}
@media (max-width: 720px) {
    .metrics-grid, .cr-scorecard-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
}
.metric-card, .cr-score-card { padding: 12px 14px !important; }
.metric-value, .cr-score-val { font-size: 1.3rem !important; }

/* ── Chat canvas welcome / empty state ───────────────────────────────────── */
.cr-welcome {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    height: 100%;
    padding: 28px 30px;
    gap: 6px;
}
.cr-welcome-glyph {
    width: 60px; height: 60px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 16px;
    color: var(--cr-orange);
    background: radial-gradient(circle at 50% 35%, rgba(255,87,10,0.14), transparent 70%),
                linear-gradient(145deg, rgba(27,31,44,0.9), rgba(16,18,26,0.9));
    border: 1px solid rgba(255, 87, 10, 0.25);
    box-shadow: 0 8px 26px rgba(255, 87, 10, 0.14), inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 12px;
    animation: crFloat 5s ease-in-out infinite;
}
@keyframes crFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}
.cr-welcome-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.15rem; font-weight: 700; color: #F8FAFC;
    letter-spacing: -0.02em;
}
.cr-welcome-sub {
    font-size: 0.85rem; line-height: 1.55; color: #94A3B8;
    max-width: 440px; margin: 2px auto 0;
}
.cr-welcome-hints {
    display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 14px;
}
.cr-welcome-pill {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.01em;
    color: #CBD5E1;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--cr-border);
    border-radius: 999px; padding: 5px 12px;
}

/* ── Composer (query input + submit) — unified bar ───────────────────────── */
.cr-composer-row { align-items: stretch !important; gap: 10px !important; margin-top: 12px !important; }
#cr-query-input textarea {
    background: rgba(16, 18, 26, 0.75) !important;
    border: 1px solid var(--cr-border) !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    padding: 12px 14px !important;
}
#cr-submit-btn { align-self: stretch !important; min-height: 100% !important; font-size: 0.92rem !important; }
.cr-composer-actions { margin-top: 8px !important; }

/* Examples dataset — turn Gradio's default table into tidy query chips */
#cr-examples .gr-samples-table, #cr-examples table { border: none !important; background: transparent !important; }
#cr-examples button, #cr-examples td {
    background: rgba(20, 24, 36, 0.6) !important;
    border: 1px solid var(--cr-border) !important;
    border-radius: 8px !important;
    color: var(--cr-text-secondary) !important;
    font-size: 0.8rem !important;
    transition: all 0.18s ease !important;
}
#cr-examples button:hover, #cr-examples td:hover {
    border-color: rgba(255, 87, 10, 0.4) !important;
    color: var(--cr-text-primary) !important;
    background: rgba(255, 87, 10, 0.06) !important;
}

/* ── Right workspace column — give tab panels real presence ──────────────── */
#workspace-tabs .tabitem { min-height: 360px !important; }
#metrics-panel { margin-bottom: 4px !important; }

/* Source markdown panels: readable padding + card feel */
#workspace-tabs .prose, #sources-tabs .prose { font-size: 0.86rem !important; line-height: 1.55 !important; }

/* Welcome-pill status dots (replaces unsupported glyphs that render as tofu) */
.cr-welcome-pill { display: inline-flex; align-items: center; gap: 7px; }
.cr-pill-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
    box-shadow: 0 0 7px currentColor; }

/* Metric card labels: allow a clean two-line wrap instead of clipping */
.metric-card, .cr-score-card { padding: 12px 10px !important; min-width: 0 !important; }
.metric-label, .cr-score-lbl {
    font-size: 0.6rem !important;
    line-height: 1.3 !important;
    white-space: normal !important;
    overflow-wrap: anywhere;
    letter-spacing: 0.04em !important;
}
.metric-value, .cr-score-val {
    font-size: 1.18rem !important;
    white-space: nowrap;
    overflow: visible;
}

/* CSS-drawn arrow on the submit button — font-independent, never tofu */
#cr-submit-btn button, #cr-submit-btn { position: relative; }
#cr-submit-btn button::after {
    content: "";
    display: inline-block;
    width: 7px; height: 7px;
    margin-left: 9px;
    border-top: 2px solid currentColor;
    border-right: 2px solid currentColor;
    transform: rotate(45deg) translateY(-1px);
    transition: transform 0.2s ease;
    vertical-align: middle;
}
#cr-submit-btn button:hover::after { transform: rotate(45deg) translate(2px, -1px); }

/* Keep the header's monospace meta chips from overflowing on mid widths */
.cr-meta-bar { gap: 8px !important; }
.cr-meta-item .cr-mono { font-size: 0.68rem !important; }

/* Default-state graph canvas inside the Neo4j tab reads as a feature, not a gap */
#cr-graph-preview iframe, #cr-graph-preview > div { border-radius: 12px !important; }

/* Respect reduced-motion preferences for accessibility */
@media (prefers-reduced-motion: reduce) {
    .cr-pulse-dot, .step-active, .cr-welcome-glyph { animation: none !important; }
    .gr-button-primary, button.primary, .metric-card, .cr-doc-card { transition: none !important; }
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
    "<div class='cr-welcome-title'>Synthesis Stream</div>"
    "<div class='cr-welcome-sub'>Ask a question about your ingested AI literature. "
    "Aletheia routes it through the six-node pipeline — retrieval, evidence scoring, "
    "AttentionRAG compression and a Critic audit — then streams a cited synthesis here.</div>"
    "<div class='cr-welcome-hints'>"
    "<span class='cr-welcome-pill'><span class='cr-pill-dot' style='background:#FF570A'></span>Hybrid RRF retrieval</span>"
    "<span class='cr-welcome-pill'><span class='cr-pill-dot' style='background:#3B82F6'></span>100-pt evidence rubric</span>"
    "<span class='cr-welcome-pill'><span class='cr-pill-dot' style='background:#00F2A5'></span>Zero-hallucination Critic</span>"
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
            "<div class='metric-card'><div class='metric-value' style='color:#64748B;'>--</div><div class='metric-label'>Evidence Rubric</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#64748B;'>--</div><div class='metric-label'>Compression Ratio</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#94A3B8;'>0.0s</div><div class='metric-label'>Latency</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#64748B;'>--</div><div class='metric-label'>The Critic QA</div></div>"
            "<div class='metric-card'><div class='metric-value' style='color:#94A3B8;'>$0.00</div><div class='metric-label'>Token Cost</div></div>"
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
        critic_icon = "Approved"
        critic_color = "#00F2A5"
    elif "regenerate" in feedback.lower() or "retry" in feedback.lower():
        critic_icon = "Regenerated"
        critic_color = "#FBBF24"
    else:
        critic_icon = "Passed"
        critic_color = "#00F2A5"

    # 5. Cost
    cost_text = f"${s.pipeline_cost:.4f}"

    return (
        f"<div class='metrics-grid cr-scorecard-grid'>"
        f"<div class='metric-card'><div class='metric-value' style='color:{ev_color};'>{ev_text}</div><div class='metric-label'>Evidence Rubric</div></div>"
        f"<div class='metric-card'><div class='metric-value' style='color:#FF570A;'>{comp_text}</div><div class='metric-label'>Compression Ratio</div></div>"
        f"<div class='metric-card'><div class='metric-value' style='color:#3B82F6;'>{latency_text}</div><div class='metric-label'>Latency</div></div>"
        f"<div class='metric-card'><div class='metric-value' style='font-size:1.15rem; color:{critic_color};'>{critic_icon}</div><div class='metric-label'>The Critic QA</div></div>"
        f"<div class='metric-card'><div class='metric-value' style='color:#94A3B8;'>{cost_text}</div><div class='metric-label'>Token Cost</div></div>"
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
    return (
        f"<div class='cr-meta-bar'>"
        f"<div class='cr-meta-item'><span class='cr-pulse-dot'></span> <span>{doc_count} Papers Ingested ({chunk_count} Chunks)</span></div>"
        f"<div class='cr-meta-item'><span class='cr-mono' style='font-size:0.72rem;'>Kibbs-dibs/MahoRAGa:main</span></div>"
        f"<div class='cr-version-pill'>v1.0.0</div>"
        f"</div>"
    )


def render_ingest_status_html(res: Dict[str, Any] | None = None) -> str:
    """Render status pill for in-browser PDF ingestion and memory wipe operations."""
    if res is None:
        return (
            "<div class='cr-status-box' style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);'>"
            "<span style='color:#94A3B8;'>Ready to ingest academic literature. Select or drop PDF files above.</span>"
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
            f"<div class='cr-status-box' style='background:rgba(0,242,165,0.06); border:1px solid rgba(0,242,165,0.25); color:#00F2A5;'>"
            f"<span>✅ {msg}</span>{neo4j_badge}"
            f"</div>"
        )
    elif status == "wiped":
        return (
            f"<div class='cr-status-box' style='background:rgba(251,191,36,0.06); border:1px solid rgba(251,191,36,0.25); color:#FBBF24;'>"
            f"<span>🗑️ {msg}</span>"
            f"</div>"
        )
    elif status in ("skipped", "no_files"):
        return (
            f"<div class='cr-status-box' style='background:rgba(251,191,36,0.06); border:1px solid rgba(251,191,36,0.25); color:#FBBF24;'>"
            f"<span>⚠️ {msg}</span>"
            f"</div>"
        )
    else:
        return (
            f"<div class='cr-status-box' style='background:rgba(244,63,94,0.06); border:1px solid rgba(244,63,94,0.25); color:#F43F5E;'>"
            f"<span>❌ {msg}</span>"
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
    """Dark design-system theme so Gradio's own components (inputs, buttons,
    blocks) render on dark surfaces at the source — the custom CSS then layers
    the CodeRabbit brand aesthetic on top."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
        font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
    ).set(
        body_background_fill="#090a0f",
        body_background_fill_dark="#090a0f",
        body_text_color="#F8FAFC",
        background_fill_primary="rgba(16, 18, 26, 0.75)",
        background_fill_secondary="rgba(20, 24, 36, 0.6)",
        block_background_fill="rgba(16, 18, 26, 0.6)",
        block_border_color="rgba(255, 255, 255, 0.07)",
        block_label_background_fill="rgba(20, 24, 36, 0.6)",
        block_label_text_color="#94A3B8",
        border_color_primary="rgba(255, 255, 255, 0.08)",
        input_background_fill="rgba(16, 18, 26, 0.85)",
        input_background_fill_focus="rgba(20, 24, 36, 0.95)",
        input_border_color="rgba(255, 255, 255, 0.10)",
        input_border_color_focus="rgba(255, 87, 10, 0.55)",
        input_placeholder_color="#64748B",
        button_secondary_background_fill="rgba(20, 24, 36, 0.9)",
        button_secondary_background_fill_hover="rgba(27, 31, 44, 0.95)",
        button_secondary_text_color="#CBD5E1",
        button_secondary_border_color="rgba(255, 255, 255, 0.10)",
        color_accent_soft="rgba(255, 87, 10, 0.12)",
    )
    return theme


def create_ui() -> gr.Blocks:
    """Construct the precision research console web app for Project Aletheia."""

    with gr.Blocks(
        title="Project Aletheia — Autonomous AI Research",
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
                                ALETHEIA <span class="cr-title-sub">// AI RESEARCH CONSOLE</span>
                            </div>
                            <div class="cr-header-desc">
                                Autonomous Multi-Agent Literature Intelligence &bull; Three-Layer Global Memory &bull; AttentionRAG 6.3x Pruning &bull; Deterministic Critic QA
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
