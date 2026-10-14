---
name: taste
description: >-
  Use when designing, building, or refining user interfaces to enforce aesthetic taste,
  typographic restraint, visual discipline, and eliminate "AI vibe-coded" clutter (gratuitous glows,
  excessive emojis, rainbow badges, fake brand gimmicks).
---

# The Taste Skill: Aesthetic Restraint & High-Craft UI Design

"Good design is as little design as possible." — Dieter Rams

AI-generated interfaces suffer from recognizable "vibe-coding" traps: chaotic gradients, random emojis, rainbow badges, aggressive glows, noisy borders, and over-complicated cards. This skill enforces intentional, disciplined design inspired by Linear, Apple, Stripe, and high-end scientific tools.

---

## 🚫 1. What to Eliminate Immediately ("Vibe-Coded" Anti-Patterns)

1. **Gratuitous Emojis Everywhere**:
   - ❌ Don't prefix every tab, heading, and button with cartoon emojis (`🧭`, `⚡`, `⚖️`, `🌐`, `✂️`, `✨`, `🐇`, `📄`, `📅`).
   - ✅ Use clean SVG icons or pure typographic labels with proper weight.
2. **Neon Glow Overload**:
   - ❌ Avoid `box-shadow: 0 0 24px rgba(255, 87, 10, 0.6)` on static cards or buttons.
   - ✅ Use diffuse, natural depth: `box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4), 0 8px 24px -4px rgba(0, 0, 0, 0.6)`.
3. **Rainbow Badge Syndrome**:
   - ❌ Don't have 5 different colored badges (orange, mint, cobalt, amber, rose) competing for attention on a single header.
   - ✅ Use a monochromatic system: crisp neutral grays with ONE singular accent color for primary actions.
4. **Fake Gimmicky Metadata**:
   - ❌ Don't add fake PR numbers or cute mascots unless explicitly part of the core product.
   - ✅ Present authoritative, factual tool status (e.g. `ALETHEIA // RESEARCH CONSOLE`, `v1.0.0`, `Memory: 3 Sources`).

---

## 📐 2. The Principles of Tasteful Design

### A. Monochromatic Foundation with Intentional Contrast
- **Canvas**: Neutral deep matte (`#0a0b0e` or `#0d0f14`), not muddy purple.
- **Surfaces**: Crisp layered dark neutrals (`#12141a`, `#171922`).
- **Dividers**: Hairline subtle borders (`1px solid rgba(255, 255, 255, 0.06)`).
- **Text Scale**:
  - `High Contrast (100%)`: `#FFFFFF` or `#F4F5F7` for headlines and key values.
  - `Subtle / Secondary (65%)`: `#9CA3AF` for body text, descriptions, and labels.
  - `Muted (40%)`: `#6B7280` for secondary metadata and units.

### B. Typographic Discipline
- Restrict to maximum 2 typefaces:
  - Sans-serif: `Inter`, `Geist`, or `Plus Jakarta Sans`.
  - Monospace: `JetBrains Mono` or `SF Mono` for data, tokens, and code.
- Letter spacing:
  - Headings: `-0.02em` to `-0.03em` for tight, editorial confidence.
  - Small uppercase labels: `0.05em` to `0.08em` for micro-typography readability.

### C. Spacing & Visual Rhythm
- Always use an 8px grid (8px, 16px, 24px, 32px).
- Generous padding inside containers; components should breathe.
- Eliminate redundant nested boxes. A container inside a card inside an accordion inside a drawer looks claustrophobic.

### D. The Tool Aesthetic (Scientific Instrument)
- Treat the UI like a precision instrument (e.g. Linear, Bloomberg, Ableton, or a CERN dashboard).
- Clear, compact, high information density without clutter.
- Responsive, crisp feedback on hover and focus.
