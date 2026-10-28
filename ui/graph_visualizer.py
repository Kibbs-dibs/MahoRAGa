"""
Interactive Knowledge Graph Visualizer for Project Aletheia (ui/graph_visualizer.py)

Generates dynamic, force-directed network graphs using Vis.js:
- Reading-room palette (ink canvas #15191c, oxblood #a23c35, brass #b79a63, sage #7fa06a)
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

# Entity type palette — reading-room bound-journal tones. Category is encoded by
# BOTH hue and shape (the shape carries meaning on its own, so the graph stays
# legible without relying on color alone).
PALETTES = {
    "Model": {"bg": "#a23c35", "border": "#c65a4f", "font": "#f7efe6", "shape": "box"},
    "Paper": {"bg": "#8a332d", "border": "#b24a41", "font": "#f7efe6", "shape": "box"},
    "Methodology": {"bg": "#2f3a40", "border": "#b79a63", "font": "#f1ece1", "shape": "ellipse"},
    "Mechanism": {"bg": "#2f3a40", "border": "#b79a63", "font": "#f1ece1", "shape": "ellipse"},
    "Benchmark": {"bg": "#3a4433", "border": "#7fa06a", "font": "#f1ece1", "shape": "diamond"},
    "Metric": {"bg": "#3a4433", "border": "#7fa06a", "font": "#f1ece1", "shape": "diamond"},
    "Concept": {"bg": "#2a2620", "border": "#b79a63", "font": "#e7dfd3", "shape": "dot"},
    "Default": {"bg": "#232a2f", "border": "#8d8676", "font": "#f1ece1", "shape": "box"},
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
                "color": {"background": style["bg"], "border": style["border"], "highlight": {"background": "#c65a4f", "border": "#f1ece1"}},
                "font": {"color": style["font"], "size": 13, "face": "Spectral, Georgia, serif"},
                "shape": style["shape"],
                "shadow": {"enabled": True, "color": "rgba(0,0,0,0.4)", "size": 5},
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
                "color": {"background": style["bg"], "border": style["border"], "highlight": {"background": "#c65a4f", "border": "#f1ece1"}},
                "font": {"color": style["font"], "size": 12, "face": "Spectral, Georgia, serif"},
                "shape": style["shape"],
                "shadow": {"enabled": True, "color": "rgba(0,0,0,0.4)", "size": 5},
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
                "color": {"color": "rgba(183, 154, 99, 0.42)", "highlight": "#a23c35", "hover": "#c65a4f"},
                "font": {"color": "#8d8676", "size": 10, "align": "middle", "background": "rgba(21, 25, 28, 0.78)", "face": "Inter, sans-serif"},
                "smooth": {"type": "curvedCW", "roundness": 0.15},
            })

    return list(nodes_map.values()), edges


def filter_graph_to_neighborhood(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    focus_entity: str,
    hops: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Scope a graph to the N-hop neighborhood of *focus_entity* (Feature 7.3).

    Enables an interactive 'zoom to this entity' filter in the graph view. Node
    identity is matched case-insensitively against node 'id' or 'label'. An unknown
    focus (or empty) returns the full graph unchanged.
    """
    if not focus_entity or not nodes:
        return nodes, edges

    focus = focus_entity.strip().lower()

    def node_key(n: Dict[str, Any]) -> str:
        return str(n.get("id", n.get("label", ""))).lower()

    matched = {node_key(n) for n in nodes if focus in node_key(n)}
    if not matched:
        return nodes, edges  # unknown focus -> no filtering

    # Expand outward `hops` times along edges.
    keep = set(matched)
    for _ in range(max(0, hops)):
        frontier = set()
        for e in edges:
            a = str(e.get("from", "")).lower()
            b = str(e.get("to", "")).lower()
            if a in keep:
                frontier.add(b)
            if b in keep:
                frontier.add(a)
        keep |= frontier

    f_nodes = [n for n in nodes if node_key(n) in keep]
    f_edges = [e for e in edges
               if str(e.get("from", "")).lower() in keep and str(e.get("to", "")).lower() in keep]
    return f_nodes, f_edges


def generate_interactive_graph_html(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    height_px: int = 420,
    title: str = "Knowledge Graph Diffusion",
) -> str:
    """
    Render a self-contained Vis.js network canvas in the reading-room palette.
    """
    if not nodes:
        return (
            f"<div style='height:{height_px}px; display:flex; flex-direction:column; align-items:center; "
            f"justify-content:center; "
            f"background:#11161a; "
            f"border:1px solid rgba(183,154,99,0.20); "
            f"border-radius:3px; color:#8d8676; font-family:Georgia,serif; text-align:center; padding:24px;'>"
            f"<div style='font-family:Spectral,Georgia,serif; font-style:italic; font-size:1.05rem; color:#cbc3b4;'>"
            f"The graph is empty</div>"
            f"<div style='font-family:Georgia,serif; font-size:0.9rem; margin-top:6px; max-width:340px; line-height:1.55;'>"
            f"Ask a question or add papers to the library, and the relationships "
            f"Aletheia finds between them will be drawn here.</div>"
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
            background: #15191c;
            color: #f1ece1;
            font-family: 'Inter', -apple-system, sans-serif;
            overflow: hidden;
        }}
        #toolbar {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 100;
            display: flex;
            gap: 10px;
            align-items: center;
            background: rgba(28, 34, 39, 0.9);
            border: 1px solid rgba(183, 154, 99, 0.20);
            border-radius: 3px;
            padding: 5px 11px;
            font-size: 0.78rem;
        }}
        .tb-badge {{
            background: transparent;
            color: #c65a4f;
            border: 1px solid rgba(162, 60, 53, 0.5);
            border-radius: 2px;
            padding: 2px 8px;
            font-weight: 500;
        }}
        .tb-btn {{
            background: transparent;
            border: 1px solid rgba(183, 154, 99, 0.20);
            color: #cbc3b4;
            border-radius: 3px;
            padding: 3px 9px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            transition: color 0.18s, border-color 0.18s;
        }}
        .tb-btn:hover {{
            background: transparent;
            color: #f1ece1;
            border-color: #b79a63;
        }}
        #legend {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            z-index: 100;
            display: flex;
            gap: 12px;
            background: rgba(28, 34, 39, 0.9);
            border: 1px solid rgba(183, 154, 99, 0.20);
            border-radius: 3px;
            padding: 5px 11px;
            font-size: 0.7rem;
            color: #8d8676;
        }}
        .lg-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .lg-dot {{
            width: 8px;
            height: 8px;
            border-radius: 1px;
        }}
        #{container_id} {{
            width: 100%;
            height: {height_px}px;
            background: #11161a;
            border: 1px solid rgba(183, 154, 99, 0.20);
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div id="toolbar">
        <span class="tb-badge">{len(nodes)} nodes</span>
        <span class="tb-badge" style="color:#b79a63; border-color:rgba(183,154,99,0.45);">{len(edges)} links</span>
        <button class="tb-btn" onclick="network.fit({{animation: true}})">Fit view</button>
        <button class="tb-btn" id="physics-btn" onclick="togglePhysics()">Freeze</button>
    </div>

    <div id="legend">
        <div class="lg-item"><span class="lg-dot" style="background:#a23c35;"></span> Model / paper</div>
        <div class="lg-item"><span class="lg-dot" style="background:#b79a63;"></span> Mechanism</div>
        <div class="lg-item"><span class="lg-dot" style="background:#7fa06a;"></span> Benchmark</div>
        <div class="lg-item"><span class="lg-dot" style="background:#8d8676;"></span> Concept</div>
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
