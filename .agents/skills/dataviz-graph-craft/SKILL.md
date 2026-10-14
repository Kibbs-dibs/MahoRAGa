---
name: dataviz-graph-craft
description: >-
  Use when creating, styling, or tuning interactive knowledge graph visualizers, Vis.js networks,
  D3 force simulations, physics parameters, node glyphs, and responsive network canvases.
---

# Data Visualization & Knowledge Graph Craft

This skill covers best practices for building high-performance, interactive, and aesthetically striking force-directed network graphs (Vis.js, D3, Cytoscape).

---

## 🕸️ 1. Vis.js Network Engine Best Practices

When embedding interactive knowledge graphs in web dashboards:
1. **Always render within an isolated `iframe`**:
   Avoid CSS conflicts with the parent container (like Gradio or React) by bundling Vis.js inside an `srcdoc` or standalone HTML file.
2. **CDN Scripts**:
   Include Vis.js Network standalone:
   ```html
   <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
   ```

---

## ⚙️ 2. Optimal Force Physics Parameters

To prevent nodes from flying off or oscillating endlessly, tune the Barnes-Hut or ForceAtlas2 solver:

```javascript
const options = {
  nodes: {
    shape: 'dot',
    size: 20,
    font: { size: 14, color: '#f8fafc', face: 'Inter, sans-serif' },
    borderWidth: 2,
    shadow: { enabled: true, color: 'rgba(0,0,0,0.5)', size: 10 }
  },
  edges: {
    width: 1.5,
    color: { color: 'rgba(255, 255, 255, 0.25)', highlight: '#FF570A' },
    arrows: { to: { enabled: true, scaleFactor: 0.7 } },
    smooth: { type: 'continuous', roundness: 0.4 }
  },
  physics: {
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {
      gravitationalConstant: -50,
      centralGravity: 0.01,
      springLength: 120,
      springConstant: 0.08,
      damping: 0.4
    },
    stabilization: { iterations: 150 }
  },
  interaction: {
    hover: true,
    navigationButtons: true,
    zoomView: true
  }
};
```

---

## 🏷️ 3. Semantic Categorization & Colors

Map entity types to semantic color tokens:
- **Methodology / Algorithm**: `#FF570A` (Coral)
- **Metric / Benchmark**: `#00F2A5` (Emerald)
- **Concept / Architecture**: `#3B82F6` (Electric Blue)
- **Document / Paper**: `#F59E0B` (Amber)
- **Default / Entity**: `#E2E8F0` (Slate)

Provide interactive UI toggles (e.g., "Freeze / Float Physics") so users can inspect node clusters without jitter.
