---
name: modern-ui-designer
description: >-
  Use when designing, building, or refining web user interfaces, design systems, CSS styles,
  color palettes, and dark mode aesthetics to produce high-end, premium interfaces (e.g. CodeRabbit, Linear, Vercel).
---

# Modern UI & Design System Craft

This skill guides the design and implementation of modern, premium, high-converting user interfaces with rich visual aesthetics, harmonious color palettes, subtle glassmorphism, and dynamic micro-animations.

---

## 🎨 1. Core Color System & Tokens

Always avoid generic raw colors (pure `#FF0000`, `#0000FF`, `#00FF00`). Use curated semantic palettes with intentional depth:

### The "Aletheia / CodeRabbit" Dark Palette
- **Canvas / Background**: `#0b0d14` (Deep obsidian void)
- **Elevated Surfaces / Cards**: `rgba(18, 22, 34, 0.75)` with `backdrop-filter: blur(16px)`
- **Border / Stroke**: `rgba(255, 255, 255, 0.08)` (Subtle crisp hairline divider)
- **Primary Accent (Brand Coral)**: `#FF570A` (High-contrast energy for CTAs, active badges, highlights)
- **Secondary Accent (Emerald Mint)**: `#00F2A5` (Success, health checks, metrics, completion)
- **Tertiary Accent (Electric Blue)**: `#3B82F6` (Knowledge nodes, links, external references)
- **Text Hierarchy**:
  - Primary: `#F8FAFC` (High contrast, crisp legibility)
  - Secondary / Muted: `#94A3B8` (Descriptions, subtitles, metadata)
  - Tertiary / Hints: `#64748B` (Placeholders, secondary metadata)

---

## 💎 2. Glassmorphism & Surface Depth

To achieve depth without visual clutter:
```css
.premium-card {
  background: rgba(18, 22, 34, 0.65);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.07);
  border-radius: 12px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.premium-card:hover {
  border-color: rgba(255, 87, 10, 0.35);
  box-shadow: 0 12px 40px 0 rgba(255, 87, 10, 0.12);
  transform: translateY(-2px);
}
```

---

## ⚡ 3. Typography & Micro-Animations

1. **Typography**:
   - Use clean, geometric sans-serif fonts: Inter, Outfit, Plus Jakarta Sans, or JetBrains Mono for code and metrics.
   - Maintain strict letter-spacing: headings (`letter-spacing: -0.02em`), uppercase labels (`letter-spacing: 0.08em; font-size: 0.75rem; font-weight: 700`).
2. **Micro-Interactions**:
   - Buttons must have active press effects: `transform: scale(0.98)` on `:active`.
   - Badges and status dots should feature soft pulsing glow effects for live activities:
     ```css
     @keyframes pulseGlow {
       0%, 100% { opacity: 1; filter: drop-shadow(0 0 6px rgba(0, 242, 165, 0.6)); }
       50% { opacity: 0.5; filter: drop-shadow(0 0 2px rgba(0, 242, 165, 0.2)); }
     }
     ```

---

## 🚫 4. Common Anti-Patterns to Avoid
- **Never use default browser gray buttons**. Always apply custom borders, backgrounds, and hover glows.
- **Never leave text without clear hierarchy**. Contrast between titles (`#F8FAFC`) and labels (`#94A3B8`) must be stark.
- **Never create boxy, sharp corners**. Use consistent rounding (`border-radius: 8px` for inputs/buttons, `12px` to `16px` for containers).
