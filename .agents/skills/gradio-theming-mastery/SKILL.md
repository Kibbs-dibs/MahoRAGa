---
name: gradio-theming-mastery
description: >-
  Use when customizing, styling, or debugging Gradio web applications, custom CSS overrides,
  Gradio layout components (Row, Column, Accordion, Tabs), custom HTML blocks, or JavaScript sidecars.
---

# Gradio Theming & Custom Component Mastery

This skill details how to override Gradio's default theme styles to create bespoke, professional web consoles without fighting the framework.

---

## 🏗️ 1. Injecting Custom CSS in Gradio

Pass custom CSS into `gr.Blocks(css=CUSTOM_CSS, theme=...)`:

```python
import gradio as gr

CUSTOM_CSS = """
/* Target Gradio container background */
.gradio-container {
    background: #0b0d14 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Custom button styling */
button.primary-btn {
    background: linear-gradient(135deg, #FF570A 0%, #E04800 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

button.primary-btn:hover {
    box-shadow: 0 4px 15px rgba(255, 87, 10, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Target Gradio Accordion headers */
.gradio-accordion .label-wrap {
    background: rgba(18, 22, 34, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
}
"""

with gr.Blocks(css=CUSTOM_CSS, title="App") as app:
    # Components...
    pass
```

---

## 📦 2. Safe Component Selectors

Gradio generates dynamic classes. To ensure style stability:
1. Always add `elem_id` or `elem_classes` to your Gradio components:
   ```python
   gr.Button("Run", elem_classes=["primary-btn"], elem_id="main-run-btn")
   ```
2. In CSS, target using:
   ```css
   #main-run-btn { ... }
   .primary-btn { ... }
   ```

---

## ⚡ 3. Real-Time Streaming & Progress

1. **`gr.Progress()`**:
   - Wrap long-running multi-agent pipelines with `progress = gr.Progress(track_tqdm=True)`.
   - Update steps dynamically: `progress(0.4, desc="Retrieving FAISS dense vectors...")`.
2. **HTML Sidecars & Dynamic Steppers**:
   - Use `gr.HTML()` to render custom status pills, pipeline stage breadcrumbs, and provenance metadata tables.
   - Return clean inline HTML strings with Tailwind-like or scoped inline CSS.
