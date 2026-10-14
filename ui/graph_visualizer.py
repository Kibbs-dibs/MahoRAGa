"""
Interactive Knowledge Graph Visualizer for Project Aletheia (ui/graph_visualizer.py)

Generates dynamic, force-directed network graphs using Vis.js:
- CodeRabbit visual theme (Dark Mauve canvas #0b0d14, Orange #FF570A, Mint #00F2A5, Cobalt #3B82F6)
- Entity categorization with distinct glyphs and colors
- Physics stabilization, zoom, pan, edge routing, and node inspection
- Seamless dual-mode: Works from Neo4j Cypher queries or parsed triple strings
"""

from __future__ import annotations

import html
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ui.graph_visualizer")

# Entity type color palettes matching CodeRabbit theme
PALETTES = {
    "Model": {"bg": "#FF570A", "border": "#FF782E", "font": "#FFFFFF", "shape": "box"},
    "Paper": {"bg": "#FF570A", "border": "#FF8A4C", "font": "#FFFFFF", "shape": "box"},
    "Methodology": {"bg": "#3B82F6", "border": "#60A5FA", "font": "#FFFFFF", "shape": "ellipse"},
    "Mechanism": {"bg": "#3B82F6", "border": "#60A5FA", "font": "#FFFFFF", "shape": "ellipse"},
    "Benchmark": {"bg": "#00F2A5", "border": "#34D399", "font": "#0B0D14", "shape": "diamond"},
    "Metric": {"bg": "#00F2A5", "border": "#34D399", "font": "#0B0D14", "shape": "diamond"},
    "Concept": {"bg": "#FBBF24", "border": "#FCD34D", "font": "#0B0D14", "shape": "dot"},
    "Default": {"bg": "#1E2235", "border": "#8E95AA", "font": "#F3F4F6", "shape": "box"},
}


def _infer_entity_type(name: str) -> str:
    """Infer semantic category from entity name heuristics."""
    low = name.lower()
    if any(k in low for k in ["rag", "model", "llama", "qwen", "gpt", "lingua", "aletheia", "transformer"]):
        return "Model"
    if any(k in low for k in ["paper", "survey", "arxiv", "et al", "conference", "ieee"]):
        return "Paper"
    if any(k in low for k in ["attention", "pruning", "retrieval", "rerank", "fusion", "embedding", "chunking"]):
        return "Mechanism"
    if any(k in low for k in ["bench", "babilong", "trivia", "hotpot", "dataset", "eval"]):
        return "Benchmark"
    if any(k in low for k in ["accuracy", "ratio", "score", "em", "latency", "f1", "fidelity"]):
        return "Metric"
    return "Concept"


def parse_triples_to_graph(triples: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse natural language triple strings into Vis.js nodes and edges.
    Example triple: 'AttentionRAG COMPRESSES Context with focal token isolation'
    """
    nodes_map: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    edge_set: Set[Tuple[str, str, str]] = set()

    for idx, raw_triple in enumerate(triples):
        # Clean triple string
        text = re.sub(r"^\[.*?\]\s*", "", raw_triple).strip()
        parts = text.split(None, 2)
        if len(parts) >= 3:
            head, relation, tail = parts[0], parts[1], parts[2]
            # Truncate very long tails for node readability
            clean_tail = tail[:40].strip()
        elif len(parts) == 2:
            head, relation, tail = parts[0], "RELATED_TO", parts[1]
            clean_tail = tail[:40].strip()
        else:
            continue

        head_clean = head.strip(" :;,.")
        tail_clean = clean_tail.strip(" :;,.")
        if not head_clean or not tail_clean:
            continue

        # Add Head Node
        if head_clean not in nodes_map:
            ent_type = _infer_entity_type(head_clean)
            style = PALETTES.get(ent_type, PALETTES["Default"])
            nodes_map[head_clean] = {
                "id": head_clean,
                "label": head_clean,
                "title": f"<b>{html.escape(head_clean)}</b><br>Type: {ent_type}",
                "color": {"background": style["bg"], "border": style["border"], "highlight": {"background": "#FF702E", "border": "#FFFFFF"}},
                "font": {"color": style["font"], "size": 13, "face": "Inter, sans-serif"},
                "shape": style["shape"],
                "shadow": {"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 6},
                "margin": 10,
            }

        # Add Tail Node
        if tail_clean not in nodes_map:
            ent_type = _infer_entity_type(tail_clean)
            style = PALETTES.get(ent_type, PALETTES["Default"])
            nodes_map[tail_clean] = {
                "id": tail_clean,
                "label": tail_clean,
                "title": f"<b>{html.escape(tail_clean)}</b><br>Type: {ent_type}",
                "color": {"background": style["bg"], "border": style["border"], "highlight": {"background": "#FF702E", "border": "#FFFFFF"}},
                "font": {"color": style["font"], "size": 12, "face": "Inter, sans-serif"},
                "shape": style["shape"],
                "shadow": {"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 6},
                "margin": 8,
            }

        edge_key = (head_clean, tail_clean, relation)
        if edge_key not in edge_set:
            edge_set.add(edge_key)
            edges.append({
                "from": head_clean,
                "to": tail_clean,
                "label": relation.lower().replace("_", " "),
                "title": f"Relation: <b>{html.escape(relation)}</b>",
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
                "color": {"color": "rgba(255, 87, 10, 0.45)", "highlight": "#FF570A", "hover": "#FF782E"},
                "font": {"color": "#9499AD", "size": 10, "align": "middle", "background": "rgba(11, 13, 20, 0.75)"},
                "smooth": {"type": "curvedCW", "roundness": 0.15},
            })

    return list(nodes_map.values()), edges


def generate_interactive_graph_html(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    height_px: int = 420,
    title: str = "Knowledge Graph Diffusion",
) -> str:
    """
    Render a self-contained Vis.js network canvas with dark CodeRabbit aesthetics.
    """
    if not nodes:
        return (
            f"<div style='height:{height_px}px; display:flex; flex-direction:column; align-items:center; "
            f"justify-content:center; background:#111420; border:1px solid rgba(255,255,255,0.08); "
            f"border-radius:12px; color:#8E95AA; font-family:Inter,sans-serif; text-align:center; padding:20px;'>"
            f"<div style='font-size:2rem; margin-bottom:8px;'>🕸️</div>"
            f"<div style='font-weight:600; font-size:0.95rem; color:#F3F4F6;'>No Active Knowledge Graph Nodes</div>"
            f"<div style='font-size:0.8rem; margin-top:4px;'>Run a literature query or ingest papers to inspect entity relationships.</div>"
            f"</div>"
        )

    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    container_id = f"vis_net_{abs(hash(nodes_json[:50]))}"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: #0b0d14;
            color: #F3F4F6;
            font-family: 'Inter', -apple-system, sans-serif;
            overflow: hidden;
        }}
        #toolbar {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 100;
            display: flex;
            gap: 6px;
            align-items: center;
            background: rgba(19, 22, 34, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 4px 10px;
            font-size: 0.76rem;
        }}
        .tb-badge {{
            background: rgba(255, 87, 10, 0.15);
            color: #FF570A;
            border: 1px solid rgba(255, 87, 10, 0.3);
            border-radius: 12px;
            padding: 2px 8px;
            font-weight: 600;
        }}
        .tb-btn {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #9499AD;
            border-radius: 6px;
            padding: 3px 8px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .tb-btn:hover {{
            background: #FF570A;
            color: #FFFFFF;
            border-color: #FF570A;
        }}
        #legend {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            z-index: 100;
            display: flex;
            gap: 8px;
            background: rgba(19, 22, 34, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 4px 10px;
            font-size: 0.7rem;
            color: #9499AD;
        }}
        .lg-item {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .lg-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
        #{container_id} {{
            width: 100%;
            height: {height_px}px;
            background: radial-gradient(circle at center, #131624 0%, #0b0d14 100%);
            border: 1px solid rgba(255, 87, 10, 0.2);
            border-radius: 12px;
        }}
    </style>
</head>
<body>
    <div id="toolbar">
        <span class="tb-badge">{len(nodes)} Nodes</span>
        <span class="tb-badge" style="background:rgba(0,242,165,0.12); color:#00F2A5; border-color:rgba(0,242,165,0.3);">{len(edges)} Edges</span>
        <button class="tb-btn" onclick="network.fit({{animation: true}})">Fit View</button>
        <button class="tb-btn" id="physics-btn" onclick="togglePhysics()">Freeze</button>
    </div>

    <div id="legend">
        <div class="lg-item"><span class="lg-dot" style="background:#FF570A;"></span> Model / Paper</div>
        <div class="lg-item"><span class="lg-dot" style="background:#3B82F6;"></span> Mechanism</div>
        <div class="lg-item"><span class="lg-dot" style="background:#00F2A5;"></span> Benchmark</div>
        <div class="lg-item"><span class="lg-dot" style="background:#FBBF24;"></span> Concept</div>
    </div>

    <div id="{container_id}"></div>

    <script type="text/javascript">
        var rawNodes = {nodes_json};
        var rawEdges = {edges_json};

        var nodes = new vis.DataSet(rawNodes);
        var edges = new vis.DataSet(rawEdges);

        var container = document.getElementById('{container_id}');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{
                borderWidth: 1.5,
                borderWidthSelected: 2.5,
                scaling: {{ min: 10, max: 30 }}
            }},
            edges: {{
                width: 1.5,
                selectionWidth: 2.5,
                hoverWidth: 2
            }},
            physics: {{
                enabled: true,
                barnesHut: {{
                    gravitationalConstant: -2200,
                    centralGravity: 0.25,
                    springLength: 95,
                    springConstant: 0.04,
                    damping: 0.09
                }},
                stabilization: {{
                    enabled: true,
                    iterations: 150,
                    updateInterval: 25
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true
            }}
        }};

        var network = new vis.Network(container, data, options);
        var physicsRunning = true;

        function togglePhysics() {{
            physicsRunning = !physicsRunning;
            network.setOptions({{ physics: {{ enabled: physicsRunning }} }});
            document.getElementById('physics-btn').innerText = physicsRunning ? 'Freeze' : 'Float';
        }}
    </script>
</body>
</html>
"""
    # Wrap in iframe for isolated, non-colliding DOM execution
    escaped_srcdoc = html.escape(html_content, quote=True)
    return (
        f"<iframe srcdoc='{escaped_srcdoc}' "
        f"style='width:100%; height:{height_px + 10}px; border:none; border-radius:12px; overflow:hidden;'></iframe>"
    )
