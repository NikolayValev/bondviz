# BondViz — Quant Terminal Redesign

**Date:** 2026-06-01
**Status:** Approved design, pending implementation plan

## Goal

Repackage the existing BondViz Streamlit app as a polished, recruiter-facing demo that
positions the author as a **front-end-capable, financially literate developer**. This is a
**presentation and structure pass**, not a feature build: it reuses all existing data
sources and computation, adds a landing page, groups navigation, applies a dark
"quant terminal" aesthetic, ensures accessibility, and makes the app easy to deploy and
share via a live URL.

A second, purely-visual design pass by the author is expected after this work — so the
aesthetic layer must be centralized and easy to tweak.

## Non-Goals

- No new data sources (Treasury XML, FRED, Polygon are the only sources).
- No new financial models or analytics.
- No migration off Streamlit in this round (a future custom front-end is noted below).
- No changes to the core math in `pricing.py` / `_native.cpp`.

## Constraints & Principles

- **Reuse existing render logic.** `visualizer.render_visualizer`, `pca_view.render_yield_pca`,
  and `stocks_view.render_polygon_stocks` keep their behavior; we wrap them in new structure
  and styling.
- **Preserve the graceful-degradation pattern.** Optional imports (`sklearn`,
  `streamlit-option-menu`, `polygon`, `pandas-datareader`) and the native-extension fallback
  must continue to degrade one feature rather than crash the app.
- **Centralize styling** so the author's later design pass touches few files.
- **Accessibility is a first-class requirement**, not an afterthought.

## Target Aesthetic

Dark "quant terminal" (Bloomberg-style): near-black background, a slightly lighter panel
color, a single accent color (amber or green), monospace numerics, dense but legible data.

## Architecture

### Navigation (native, grouped)

Replace the current `streamlit-option-menu` header bar with Streamlit's native
`st.navigation` / `st.Page` (available in the pinned `streamlit>=1.38`). It renders a grouped
sidebar automatically and is keyboard- and screen-reader-accessible.

Group structure:

- **Home** → `home_view.render_home`
- **Fixed Income**
  - Yield Curve → `visualizer.render_visualizer`
  - Bond Pricing → `pricing_view.render_pricing` (new; extracted from `streamlit_app.py`)
  - PCA Factors → `pca_view.render_yield_pca`
- **Markets**
  - Stocks → `stocks_view.render_polygon_stocks`

`app/streamlit_app.py` becomes a thin entry point: set page config, apply theme/CSS once,
build the grouped `st.navigation(...)` mapping, and run the selected page.

### New / changed modules

| Module | Responsibility | Depends on |
| --- | --- | --- |
| `app/streamlit_app.py` | Entry point: page config, theme bootstrap, grouped `st.navigation`, dispatch | `theme`, all `*_view` modules |
| `src/bondviz/home_view.py` (new) | Landing page: hero, live KPI cards, grouped nav cards | `app_logic`, `treasury`, `theme` |
| `src/bondviz/pricing_view.py` (new) | Bond PV calculator page (extracted from current "Curve Calculations") | `app_logic`, `theme` |
| `src/bondviz/theme.py` (new) | `inject_global_css()`, `apply_mpl_style()`, small UI helpers (card/section wrappers, KPI metric) | `streamlit`, `matplotlib` |
| `src/bondviz/visualizer.py` | Unchanged behavior; reuse `theme` helpers instead of its local print-CSS | `theme` |
| `src/bondviz/pca_view.py` | Unchanged behavior; charts pick up `apply_mpl_style()` | `theme` |
| `src/bondviz/stocks_view.py` | Unchanged behavior; add clear "set POLYGON_API_KEY in secrets" message on missing key | `theme` |
| `.streamlit/config.toml` (new) | Dark `[theme]` palette + monospace font | — |
| `requirements.txt` (new) | Runtime deps for Streamlit Community Cloud (pure-Python; no build step) | — |

The existing print-CSS block in `visualizer.py` is generalized into `theme.inject_global_css()`
so all pages share one source of styling (this is targeted cleanup in code we're touching, not
a broad refactor).

### Home / front page (`home_view.py`)

- **Hero:** "BONDVIZ" wordmark, one-line positioning tagline, a short "what this is" blurb.
- **Live KPI cards:** values pulled from data already fetched — latest 10Y, 2s10s spread,
  3m10y spread — via existing `app_logic` / `treasury` helpers. Wrapped in
  `@st.cache_data` + try/except; on a feed hiccup, show a friendly placeholder instead of
  crashing.
- **Grouped navigation cards:** `st.page_link` cards into Fixed Income and Markets.

### Theme layer (`theme.py`)

- `inject_global_css()` — terminal cards, panel borders, metric styling, accent treatment;
  idempotent via a `st.session_state` flag (same guard pattern as today's print-CSS).
- `apply_mpl_style()` — dark matplotlib style (dark figure/axes background, light text/grid,
  accent line color) applied centrally so **every** chart across all pages is cohesive.
- Small helpers: a section/card context manager and a KPI metric renderer.

## Accessibility

- WCAG-AA contrast for text and accents against the dark palette.
- No color-only signaling: spreads/curve shape keep +/- signs and text labels (already done);
  charts keep captions and interpretation text.
- Semantic headings and chart captions; native nav provides keyboard/screen-reader support.
- Readable base font size.

## Deployability ("easy to share")

Target: **Streamlit Community Cloud**, so the author can hand recruiters a live URL.

- `requirements.txt` lists runtime dependencies only (mirroring `pyproject.toml` deps) and
  does **not** install the package with `pip install -e .`, so the cloud build needs **no C++
  toolchain**. The app already falls back to pure-Python pricing when `bondviz._native` is
  absent.
- Entry point for the cloud app: `app/streamlit_app.py`. Because the package is not pip-
  installed on cloud, the entry point prepends the repo's `src/` to `sys.path` before importing
  `bondviz` (a small, documented shim at the top of the file). This keeps both local
  (`PYTHONPATH=src`, already used by the `.bat` scripts) and cloud runs working with no build
  step.
- **No-secret first run:** Home, Yield Curve, Bond Pricing, and PCA (Treasury + FRED) work
  with zero configuration. The **Stocks** page degrades gracefully when `POLYGON_API_KEY` is
  unset, showing a clear "add your key in Streamlit secrets" message.
- Deployment steps documented in `README.md`.

## Future Front-End (next round, not in scope)

Documented migration path only: a custom HTML/CSS/JS or React front-end that calls the same
`bondviz` data/compute functions (`treasury`, `pricing`, `app_logic`, `stocks`) behind a thin
API or pre-generated data. The current refactor keeps data/compute cleanly separated from
the Streamlit view layer to make that migration straightforward.

## Risks & Mitigations

- **`src/` import on Streamlit Cloud** — handled by the `sys.path` shim in
  `app/streamlit_app.py` (no package install, no build).
- **Native extension build on cloud** — avoided by shipping a pure-Python `requirements.txt`;
  the fallback path is already exercised.
- **Live KPI fetch latency/failure on Home** — cached + try/except with placeholder.
- **Matplotlib dark style vs. Streamlit theme drift** — both driven from `theme.py` /
  `config.toml`; verify visually after wiring.

## Success Criteria

- Grouped sidebar nav (Home / Fixed Income / Markets) renders and is keyboard-navigable.
- A landing page exists with hero, live KPIs (or graceful placeholder), and nav cards.
- All pages and all charts share the dark quant-terminal aesthetic from one styling source.
- App runs end-to-end with **no** native extension and **no** Polygon key (Stocks shows a
  clear setup message).
- App deploys to Streamlit Community Cloud from a clean checkout and is reachable by URL.
