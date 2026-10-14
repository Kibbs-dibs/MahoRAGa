---
name: accessibility-and-responsive
description: >-
  Use when optimizing web layouts for mobile/desktop responsiveness, checking color contrast ratios,
  auditing accessibility standards (WCAG 2.1 AA), or preventing layout shifts (CLS).
---

# Accessibility & Responsive Web Engineering

This skill provides guidelines and checklists to ensure web interfaces are responsive, accessible, and compliant with modern UX standards.

---

## 📱 1. Mobile & Desktop Responsive Design

1. **Fluid Grid Layouts**:
   - Use CSS Grid and Flexbox with `minmax()` and `auto-fit` rather than hardcoded pixel widths.
   - Example:
     ```css
     .responsive-grid {
       display: grid;
       grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
       gap: 16px;
     }
     ```
2. **Breakpoints**:
   - Mobile: `< 640px` (Single column, collapsible sidebars, touch-friendly tap targets `min-height: 44px`).
   - Tablet: `640px - 1024px` (Two columns, compact tables).
   - Desktop: `> 1024px` (Full layout with persistent navigation and docks).

---

## 👁️ 2. WCAG 2.1 AA Contrast Ratios

- **Normal Text (< 18pt)**: Requires at least **4.5:1** contrast ratio against the background.
  - On `#0b0d14` background:
    - ✅ `#F8FAFC` provides **16.8:1** (Superb)
    - ✅ `#94A3B8` provides **8.3:1** (Passes AA & AAA)
    - ❌ `#334155` provides **2.3:1** (Fails — do not use for readable text)
- **Large Text (≥ 18pt or 14pt bold)**: Requires at least **3.0:1** contrast ratio.
- **Interactive Controls (Inputs, Buttons, Focus Rings)**: Requires **3.0:1** contrast against adjacent backgrounds.

---

## ⌨️ 3. Keyboard & Screen Reader Accessibility

1. **Focus States**: Never remove focus rings without providing a customized visible focus indicator:
   ```css
   button:focus-visible, input:focus-visible {
     outline: 2px solid #FF570A;
     outline-offset: 2px;
   }
   ```
2. **Semantic Elements & ARIA**:
   - Use `<nav>`, `<header>`, `<main>`, `<article>`, and `<section>`.
   - Ensure dynamic asynchronous loaders have `role="status"` and `aria-live="polite"`.
