# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Setup, run, and dev helpers are Windows `.bat` wrappers in `scripts/win/` (Unix equivalents in `scripts/unix/`). They all `cd` to the repo root, require `.venv\Scripts\python.exe`, and export `PYTHONPATH=%CD%\src` before invoking Python.

- **Setup venv + install:** `scripts\win\setup_venv.bat` (creates `.venv`, `pip install -e .`, then installs app deps). Equivalent manual: `python -m venv .venv` then `.venv\Scripts\python.exe -m pip install -e .`
- **Run Streamlit app:** `scripts\win\run_streamlit.bat` (or `python -m streamlit run app\streamlit_app.py`). Serves on http://localhost:8501.
- **CLI – fetch latest yield curve:** `scripts\win\run_cli_fetch.bat` → saves `yield_curve.png`
- **CLI – price a bond:** `scripts\win\run_cli_price.bat` (example args: `--coupon 0.05 --yield 0.04 --years 10 --face 1000`)
- **Sanity check interpreter + import:** `scripts\win\dev_check.bat`
- **Run tests:** `.venv\Scripts\python.exe -m pytest`
- **Cloud deps:** `requirements.txt` lists runtime deps for Streamlit Community Cloud (pure Python, no C++ build); `pyproject.toml` is the source of truth for local installs.

Tests live in `tests/` and cover only pure logic (KPI computation, theme rcParams); the
Streamlit UI is verified by running the app. There is no linter/formatter configured.

## Native extension

`bondviz._native` is a pybind11 C++ extension built from `src/bondviz/_native.cpp` via `setup.py`. It is marked `optional=True`, so a failed C++ build does **not** break `pip install` — the app silently falls back to pure-Python math. Requires a C++17 toolchain (MSVC Build Tools on Windows). `pricing.py` always tries `_native` first and falls back on any exception, so the C++ and Python implementations of `pv_continuous` / `discount_factors_continuous` must stay numerically identical.

## Architecture

Two-part layout: an importable library under `src/bondviz/` and thin entry points (`app/streamlit_app.py`, `scripts/*.py`) that import it.

**Data sources** (three, independent):
- `treasury.py` — U.S. Treasury daily par-yield XML feed, no API key. Robust to XML namespace changes and missing date tags. Columns come back as `BC_*` (e.g. `BC_10YEAR`).
- FRED — loaded via `pandas-datareader` inside `visualizer.py` (`load_from_fred`), used as fallback to the Treasury source.
- `stocks.py` — Polygon.io via `polygon-api-client`. Key resolution order: `api_key` arg → `POLYGON_API_KEY` env (also reads `.env` via `python-dotenv`) → `st.secrets`.

**Tenor canonicalization** lives in `visualizer.py`: `TENOR_ORDER`, `TENOR_YEARS`, `TENOR_TO_SERIES` (FRED codes), and `BC_TO_TENOR` (maps Treasury `BC_*` → canonical labels like `1M`/`10Y`). Any code mixing Treasury and FRED data must go through these maps.

**Streamlit app** (`app/streamlit_app.py`) is a thin entry point: it prepends `src/` to
`sys.path` (so `bondviz` imports without a pip install — this is what makes Streamlit Cloud
work with a build-free `requirements.txt`), applies the theme, then builds a grouped
`st.navigation` of `st.Page` objects. Each page is one render function:

- **Home** → `home_view.render_home(pages)` — hero, live KPI snapshot (10Y, 2s10s, 3m10y via
  `app_logic.compute_curve_kpis`, cached + fail-soft), and `st.page_link` cards into the groups.
- **Fixed Income**: *Yield Curve* → `visualizer.render_visualizer()` (curve, shifts, spreads,
  heatmap, semiannual par→zero/forward bootstrap); *Bond Pricing* → `pricing_view.render_pricing()`
  (continuous-compounding PV); *PCA Factors* → `pca_view.render_yield_pca()`.
- **Markets**: *Stocks* → `stocks_view.render_polygon_stocks()` (degrades to a "set
  POLYGON_API_KEY in secrets" message when no key is present).

A name→`st.Page` registry built in the entry point is passed to `render_home` so Home's cards
can link to other pages. **All styling is centralized in `theme.py`** (`inject_global_css`,
`apply_mpl_style`, `card`, `kpi`) plus `.streamlit/config.toml`; `visualizer.py` aliases its old
`_inject_print_styles`/`_print_section` names to the theme helpers. Edit the look there, in one
place. `apply_mpl_style()` runs once at startup so every matplotlib chart renders dark.

**Cross-page state:** `render_visualizer()` writes the loaded yield DataFrame to `st.session_state["yield_curve_df"]`; `pca_view.py` reuses it (the "use data loaded in Treasury Yields page" option) so PCA runs on whatever the user loaded. Heavy loads/PCA are wrapped in `@st.cache_data`/`@st.cache_resource`.

**Bond math conventions:** the core pricing model (`pricing.py`, `_native.cpp`) uses **continuous compounding**. The visualizer's bootstrapping (`bootstrap_zeros_from_par`) separately assumes **semiannual** par coupons and converts to annual-compounded zero rates — don't conflate the two compounding conventions when editing.

**Optional-dependency pattern:** `scikit-learn`, `streamlit-option-menu`, `polygon`, and `pandas-datareader` are all imported lazily inside try/except so missing packages degrade one feature rather than crashing the app. Preserve this when adding feature code.
