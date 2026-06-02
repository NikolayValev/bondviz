# BondViz Quant-Terminal Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repackage the existing BondViz Streamlit app as a recruiter-facing demo with a landing page, grouped native navigation, a centralized dark "quant terminal" theme, accessibility, and one-click Streamlit Community Cloud deployability — reusing all existing data and compute.

**Architecture:** Keep all existing data/compute (`treasury`, `pricing`, `app_logic`, `stocks`, `pca_view`, `visualizer`). Add a thin `st.navigation` entry point that groups pages (Home / Fixed Income / Markets), a `theme.py` module that is the single source of styling for both CSS and matplotlib, a `home_view.py` landing page driven by already-fetched Treasury data, and a `pricing_view.py` extracted from the old combined page. Deployability comes from a pure-Python `requirements.txt` plus a `sys.path` shim so no C++ build is needed.

**Tech Stack:** Python ≥3.10, Streamlit ≥1.38 (`st.navigation`/`st.Page`/`st.page_link`), pandas, matplotlib, pytest (new, dev-only).

---

## File Structure

| File | Responsibility | Status |
| --- | --- | --- |
| `src/bondviz/theme.py` | Single source of the dark aesthetic: palette constants, `apply_mpl_style()`, `inject_global_css()`, `card()` context manager, `kpi()` helper | Create |
| `src/bondviz/app_logic.py` | Add `compute_curve_kpis()` (pure function over a Treasury yield row) | Modify |
| `src/bondviz/home_view.py` | Landing page: hero, live KPI cards, grouped nav cards | Create |
| `src/bondviz/pricing_view.py` | Bond PV calculator page (extracted from `app/streamlit_app.py`) | Create |
| `app/streamlit_app.py` | Thin entry point: `sys.path` shim, page config, theme bootstrap, grouped `st.navigation`, page registry | Rewrite |
| `src/bondviz/visualizer.py` | Reuse `theme` helpers instead of its local print-CSS | Modify |
| `src/bondviz/stocks_view.py` | Inject theme CSS; show clear "set POLYGON_API_KEY" message when key is missing | Modify |
| `.streamlit/config.toml` | Dark `[theme]` palette | Create |
| `requirements.txt` | Runtime deps for Streamlit Cloud (no `-e .`, no build step) | Create |
| `pyproject.toml` | Add `dev` optional-dependency (`pytest`) | Modify |
| `tests/` | Unit tests for the pure logic (`compute_curve_kpis`, `apply_mpl_style`) | Create |
| `README.md` | Add deployment + new structure docs | Modify |
| `CLAUDE.md` | Update commands/architecture for the new layout | Modify |

**Testing approach:** This is a UI/presentation refactor. Only genuinely pure logic gets unit tests (KPI computation, matplotlib style). Visual/layout work is verified by running the app (Task 11). This is honest — Streamlit layout cannot be meaningfully unit-tested.

---

## Task 1: Dev test harness

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add a dev optional-dependency for pytest**

In `pyproject.toml`, after the `[project]` `dependencies = [...]` list (before `[tool.setuptools]`), add:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: Create the tests package**

Create `tests/__init__.py` (empty file).

- [ ] **Step 3: Make `bondviz` importable in tests without install**

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

- [ ] **Step 4: Install pytest into the venv**

Run: `.venv\Scripts\python.exe -m pip install pytest`
Expected: installs successfully.

- [ ] **Step 5: Verify pytest collects nothing yet (no error)**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: `no tests ran` (exit code 5 is acceptable here — it means zero tests collected, not a failure).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/__init__.py tests/conftest.py
git commit -m "test: add pytest dev harness"
```

---

## Task 2: `compute_curve_kpis` (pure logic, TDD)

**Files:**
- Test: `tests/test_app_logic.py`
- Modify: `src/bondviz/app_logic.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_app_logic.py`:

```python
import pandas as pd

from bondviz.app_logic import compute_curve_kpis


def test_compute_curve_kpis_basic():
    row = pd.Series({"BC_3MONTH": 5.0, "BC_2YEAR": 4.0, "BC_10YEAR": 4.5})
    k = compute_curve_kpis(row)
    assert k["10Y"] == 4.5
    assert k["2s10s"] == 0.5      # 10Y - 2Y, in percentage points
    assert k["3m10y"] == -0.5     # 10Y - 3M


def test_compute_curve_kpis_missing_returns_none():
    row = pd.Series({"BC_10YEAR": 4.5})
    k = compute_curve_kpis(row)
    assert k["10Y"] == 4.5
    assert k["2s10s"] is None
    assert k["3m10y"] is None


def test_compute_curve_kpis_handles_nan():
    row = pd.Series({"BC_2YEAR": float("nan"), "BC_10YEAR": 4.5})
    k = compute_curve_kpis(row)
    assert k["2s10s"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_app_logic.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_curve_kpis'`.

- [ ] **Step 3: Implement the function**

In `src/bondviz/app_logic.py`, add at the end of the file:

```python
def compute_curve_kpis(latest: pd.Series) -> dict[str, float | None]:
    """Headline yield-curve KPIs from a Treasury par-yield row (BC_* columns, percent).

    Returns 10Y level plus the 2s10s and 3m10y slopes in percentage points.
    Any missing/NaN input yields None for the affected metric.
    """
    def _val(col: str) -> float | None:
        v = latest.get(col)
        return float(v) if v is not None and pd.notna(v) else None

    y10 = _val("BC_10YEAR")
    y2 = _val("BC_2YEAR")
    y3m = _val("BC_3MONTH")
    return {
        "10Y": y10,
        "2s10s": (y10 - y2) if (y10 is not None and y2 is not None) else None,
        "3m10y": (y10 - y3m) if (y10 is not None and y3m is not None) else None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_app_logic.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bondviz/app_logic.py tests/test_app_logic.py
git commit -m "feat: add compute_curve_kpis for home snapshot"
```

---

## Task 3: Theme module (`theme.py`)

**Files:**
- Create: `src/bondviz/theme.py`
- Test: `tests/test_theme.py`

- [ ] **Step 1: Write the failing test for the matplotlib style**

Create `tests/test_theme.py`:

```python
import matplotlib as mpl

from bondviz import theme


def test_apply_mpl_style_sets_dark_background():
    mpl.rcParams["figure.facecolor"] = "white"  # ensure a known starting point
    theme.apply_mpl_style()
    assert mpl.rcParams["figure.facecolor"] == theme.BG
    assert mpl.rcParams["axes.facecolor"] == theme.PANEL
    assert mpl.rcParams["axes.grid"] is True


def test_palette_constants_exist():
    for name in ("BG", "PANEL", "ACCENT", "TEXT", "MUTED"):
        assert getattr(theme, name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_theme.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bondviz.theme'`.

- [ ] **Step 3: Implement `theme.py`**

Create `src/bondviz/theme.py`:

```python
from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import streamlit as st

# --- Palette: single source of truth for the dark quant-terminal look ---------
BG = "#0a0e14"            # app background
PANEL = "#131722"         # cards / chart axes
PANEL_BORDER = "rgba(255, 255, 255, 0.08)"
ACCENT = "#00d68f"        # primary accent (green), AA-contrast on BG
TEXT = "#e6e6e6"          # body text, AA-contrast on BG/PANEL
MUTED = "#8b95a7"         # secondary text / ticks
GRID = "rgba(255, 255, 255, 0.10)"

_CSS_FLAG = "_bondviz_css_loaded"
CARD_CLASS = "bv-card"


def apply_mpl_style() -> None:
    """Apply the dark chart style globally so every matplotlib figure matches."""
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linestyle": ":",
        "legend.facecolor": PANEL,
        "legend.edgecolor": MUTED,
        "legend.labelcolor": TEXT,
        "figure.autolayout": True,
        "axes.prop_cycle": mpl.cycler(
            color=[ACCENT, "#5b8def", "#f5a623", "#e5484d", "#9b59b6"]
        ),
    })


def inject_global_css() -> None:
    """Inject the terminal CSS once per session (idempotent)."""
    if st.session_state.get(_CSS_FLAG):
        return
    st.session_state[_CSS_FLAG] = True
    st.markdown(
        f"""
        <style>
        /* Cards / panels */
        .{CARD_CLASS} {{
            background: {PANEL};
            border: 1px solid {PANEL_BORDER};
            border-left: 3px solid {ACCENT};
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.1rem;
        }}
        /* Monospace numerics for the terminal feel (body text stays sans for readability) */
        [data-testid="stMetricValue"],
        [data-testid="stDataFrame"] {{
            font-variant-numeric: tabular-nums;
            font-family: "SFMono-Regular", "Consolas", "Roboto Mono", monospace;
        }}
        [data-testid="stMetricValue"] {{ color: {ACCENT}; }}
        h1, h2, h3 {{ letter-spacing: 0.01em; }}

        /* Print to PDF: hide chrome, keep cards from splitting (preserves prior behavior) */
        @media print {{
            header, footer,
            [data-testid="stSidebar"],
            [data-testid="stToolbar"] {{ display: none !important; }}
            .{CARD_CLASS} {{ break-inside: avoid; page-break-inside: avoid; }}
            figure, img, canvas {{ break-inside: avoid; page-break-inside: avoid; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def card():
    """Wrap content in a terminal-style card."""
    st.markdown(f'<div class="{CARD_CLASS}">', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


def kpi(label: str, value: str, help_text: str | None = None) -> None:
    """Render a single KPI metric."""
    st.metric(label, value, help=help_text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_theme.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bondviz/theme.py tests/test_theme.py
git commit -m "feat: add centralized dark theme module"
```

---

## Task 4: Streamlit theme config

**Files:**
- Create: `.streamlit/config.toml`

- [ ] **Step 1: Create the config**

Create `.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#00d68f"
backgroundColor = "#0a0e14"
secondaryBackgroundColor = "#131722"
textColor = "#e6e6e6"

[server]
runOnSave = true
```

- [ ] **Step 2: Commit**

```bash
git add .streamlit/config.toml
git commit -m "feat: add dark Streamlit theme config"
```

Note: colors mirror `theme.py` constants. If you change one, change both (verified visually in Task 11).

---

## Task 5: Bond Pricing page (`pricing_view.py`)

**Files:**
- Create: `src/bondviz/pricing_view.py`

- [ ] **Step 1: Implement the page (extracted from the old combined page)**

Create `src/bondviz/pricing_view.py`:

```python
from __future__ import annotations

import streamlit as st

from . import theme
from .app_logic import compute_pv


def render_pricing() -> None:
    theme.inject_global_css()
    st.header("Bond Pricing")
    st.caption("Present value of a fixed-coupon bond under continuous compounding.")

    with st.sidebar.form("pv_form"):
        st.subheader("Inputs")
        face = st.number_input("Face", value=1000.0, step=100.0, key="pv_face")
        coupon = st.number_input(
            "Coupon rate", value=0.05, step=0.005, format="%.3f", key="pv_coupon"
        )
        ytm = st.number_input(
            "Continuous yield", value=0.04, step=0.005, format="%.3f", key="pv_yield"
        )
        years = st.number_input("Years to maturity", value=10.0, step=1.0, key="pv_years")
        submitted = st.form_submit_button("Calculate")

    if submitted or "pv_result" not in st.session_state:
        st.session_state["pv_result"] = compute_pv(face, coupon, ytm, years)

    with theme.card():
        st.metric("Present Value", f"{st.session_state['pv_result']:,.2f}")
        st.caption(
            f"Face {face:,.0f} · coupon {coupon:.3%} · cont. yield {ytm:.3%} · {years:g}y"
        )
```

- [ ] **Step 2: Verify it imports**

Run: `.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from bondviz.pricing_view import render_pricing; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/bondviz/pricing_view.py
git commit -m "feat: extract Bond Pricing into its own page"
```

---

## Task 6: Home / landing page (`home_view.py`)

**Files:**
- Create: `src/bondviz/home_view.py`

- [ ] **Step 1: Implement the landing page**

Create `src/bondviz/home_view.py`:

```python
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import theme
from .app_logic import compute_curve_kpis
from .treasury import latest_par_yields


@st.cache_data(show_spinner=False, ttl=3600)
def _safe_latest_kpis():
    """Fetch latest Treasury KPIs; degrade to (None, None) on any failure."""
    try:
        latest = latest_par_yields()
        obs = pd.to_datetime(latest["DATE"]).date()
        return compute_curve_kpis(latest), obs
    except Exception:
        return None, None


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}%"


def _fmt_bps(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:+.0f} bps"


def render_home(pages: dict | None = None) -> None:
    theme.inject_global_css()

    st.markdown("# BONDVIZ")
    st.markdown("##### Fixed-income & markets research terminal")
    st.write(
        "BondViz pulls live U.S. Treasury and market data to visualize the yield curve, "
        "price bonds, decompose curve moves with PCA, and chart equities. Built as a "
        "front-end-focused demo by a financially literate developer."
    )

    kpis, obs = _safe_latest_kpis()
    with theme.card():
        st.subheader("Snapshot" + (f" · {obs}" if obs else ""))
        if kpis is None:
            st.info(
                "Live Treasury snapshot is unavailable right now — "
                "explore the pages from the sidebar."
            )
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                theme.kpi("10Y Treasury", _fmt_pct(kpis["10Y"]))
            with c2:
                theme.kpi("2s10s spread", _fmt_bps(kpis["2s10s"]))
            with c3:
                theme.kpi("3m10y spread", _fmt_bps(kpis["3m10y"]))

    st.subheader("Explore")
    g1, g2 = st.columns(2)
    with g1:
        with theme.card():
            st.markdown("**Fixed Income**")
            if pages:
                st.page_link(pages["Yield Curve"], label="Yield Curve", icon="📈")
                st.page_link(pages["Bond Pricing"], label="Bond Pricing", icon="🧮")
                st.page_link(pages["PCA Factors"], label="PCA Factors", icon="🧬")
    with g2:
        with theme.card():
            st.markdown("**Markets**")
            if pages:
                st.page_link(pages["Stocks"], label="Stocks", icon="📊")
```

- [ ] **Step 2: Verify it imports**

Run: `.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from bondviz.home_view import render_home; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/bondviz/home_view.py
git commit -m "feat: add home landing page with live KPI snapshot"
```

---

## Task 7: Rewire entry point to grouped navigation

**Files:**
- Rewrite: `app/streamlit_app.py`

- [ ] **Step 1: Replace the entry point**

Replace the entire contents of `app/streamlit_app.py` with:

```python
import sys
from pathlib import Path

# Make `bondviz` importable when the package isn't pip-installed (e.g. Streamlit Cloud).
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from bondviz import theme  # noqa: E402
from bondviz.home_view import render_home  # noqa: E402
from bondviz.pca_view import render_yield_pca  # noqa: E402
from bondviz.pricing_view import render_pricing  # noqa: E402
from bondviz.stocks_view import render_polygon_stocks  # noqa: E402
from bondviz.visualizer import render_visualizer  # noqa: E402

st.set_page_config(page_title="BondViz", page_icon="📈", layout="wide")
theme.apply_mpl_style()
theme.inject_global_css()

# Pages (functions wrapped as st.Page) + a name->page registry for cross-page links.
yield_curve = st.Page(render_visualizer, title="Yield Curve", icon="📈", url_path="yield-curve")
bond_pricing = st.Page(render_pricing, title="Bond Pricing", icon="🧮", url_path="bond-pricing")
pca_factors = st.Page(render_yield_pca, title="PCA Factors", icon="🧬", url_path="pca")
stocks = st.Page(render_polygon_stocks, title="Stocks", icon="📊", url_path="stocks")

_registry = {
    "Yield Curve": yield_curve,
    "Bond Pricing": bond_pricing,
    "PCA Factors": pca_factors,
    "Stocks": stocks,
}

home = st.Page(
    lambda: render_home(_registry),
    title="Home",
    icon="🏠",
    url_path="home",
    default=True,
)

nav = st.navigation(
    {
        "": [home],
        "Fixed Income": [yield_curve, bond_pricing, pca_factors],
        "Markets": [stocks],
    }
)
nav.run()
```

- [ ] **Step 2: Smoke-test the app boots without runtime errors**

Run (headless, then stop after a few seconds):
`.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.headless true`
Expected: console shows "You can now view your Streamlit app" with no traceback. Stop with Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: grouped st.navigation (Home / Fixed Income / Markets)"
```

---

## Task 8: Route visualizer styling through `theme`

**Files:**
- Modify: `src/bondviz/visualizer.py:43-103`

- [ ] **Step 1: Replace the local print-CSS plumbing with theme imports**

In `src/bondviz/visualizer.py`, delete the block defining `PRINT_STYLE_SESSION_KEY`, `PRINT_SECTION_CLASS`, `_inject_print_styles()`, and `_print_section()` (lines 43–103, from `PRINT_STYLE_SESSION_KEY = ...` through the end of the `_print_section` context manager). Replace it with:

```python
# Styling is centralized in theme.py; keep the original call-site names as aliases.
from .theme import inject_global_css as _inject_print_styles  # noqa: E402
from .theme import card as _print_section  # noqa: E402
```

This preserves every existing `_inject_print_styles()` and `with _print_section():` call site unchanged.

- [ ] **Step 2: Verify it imports and call sites still resolve**

Run: `.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); import bondviz.visualizer as v; print(v._inject_print_styles, v._print_section)"`
Expected: prints two callables, no error.

- [ ] **Step 3: Commit**

```bash
git add src/bondviz/visualizer.py
git commit -m "refactor: route visualizer styling through theme module"
```

---

## Task 9: Stocks page — theme + missing-key message

**Files:**
- Modify: `src/bondviz/stocks_view.py:1-11`, `src/bondviz/stocks_view.py:29-30`

- [ ] **Step 1: Import theme**

In `src/bondviz/stocks_view.py`, add to the relative-import section (after `from .stocks import fetch_aggregates`):

```python
from . import theme
```

- [ ] **Step 2: Add CSS + graceful missing-key guard**

In `render_polygon_stocks()`, replace the first line `st.header("Stocks (Polygon.io)")` with:

```python
    theme.inject_global_css()
    st.header("Stocks (Polygon.io)")
    if not _default_polygon_key():
        st.info(
            "**Polygon API key not set.** Add `POLYGON_API_KEY` to your Streamlit secrets "
            "(Manage app → Settings → Secrets) or a local `.env` to enable live stock data. "
            "The Home and Fixed Income pages work without any key."
        )
        return
```

- [ ] **Step 3: Verify it imports**

Run: `.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from bondviz.stocks_view import render_polygon_stocks; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/bondviz/stocks_view.py
git commit -m "feat: theme + clear missing-key message on Stocks page"
```

---

## Task 10: Deployment artifacts + docs

**Files:**
- Create: `requirements.txt`
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Create `requirements.txt` (runtime only, no build step)**

Create `requirements.txt`:

```text
requests>=2.32.0
pandas>=2.2.2
pandas-datareader>=0.10.0
python-dotenv>=1.0.1
matplotlib>=3.9.0
streamlit>=1.38.0
polygon-api-client>=1.13.0
streamlit-option-menu>=0.3.6
scikit-learn>=1.3.0
```

- [ ] **Step 2: Add a Deployment section to `README.md`**

Append to `README.md`:

```markdown
## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On https://share.streamlit.io, create a new app pointing at `app/streamlit_app.py`.
3. Streamlit installs `requirements.txt` (pure Python — no C++ build; bond math falls
   back to the Python implementation when the native extension is absent).
4. Home and Fixed Income work with no configuration. For the Stocks page, add a secret
   under **Manage app → Settings → Secrets**:

   ```toml
   POLYGON_API_KEY = "your_key_here"
   ```
```

- [ ] **Step 3: Update `CLAUDE.md` for the new layout**

In `CLAUDE.md`, update the **Architecture → Streamlit app** subsection to reflect grouped
`st.navigation` (Home / Fixed Income / Markets), the new `home_view.py`, `pricing_view.py`,
and `theme.py` modules, and the `sys.path` shim in `app/streamlit_app.py`. Update the
**Commands** section to note `requirements.txt` (cloud) and `python -m pytest` (tests).
Replace the "no test suite" sentence with: "Tests live in `tests/`; run
`.venv\Scripts\python.exe -m pytest`. Only pure logic is covered (KPIs, theme); UI is
verified by running the app."

- [ ] **Step 4: Commit**

```bash
git add requirements.txt README.md CLAUDE.md
git commit -m "docs: add deployment artifacts and update docs for redesign"
```

---

## Task 11: Full manual verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Launch the app**

Run: `.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py`
Open http://localhost:8501.

- [ ] **Step 3: Verify navigation and pages**

Confirm:
- Sidebar shows groups: Home (top), **Fixed Income** (Yield Curve, Bond Pricing, PCA Factors), **Markets** (Stocks).
- Home shows hero, a Snapshot card with 3 KPIs (or a graceful info message), and Explore cards whose links switch pages.
- Each page renders with the dark theme; **matplotlib charts have dark backgrounds** matching the app.
- Bond Pricing computes a PV. Yield Curve and PCA render charts. Stocks shows the missing-key message if no `POLYGON_API_KEY`, otherwise fetches.

- [ ] **Step 4: Accessibility spot-check**

Confirm: text is readable against the dark background (no low-contrast gray-on-dark), spread KPIs show +/- signs (not color-only), and you can Tab through the sidebar nav with the keyboard.

- [ ] **Step 5: Final commit (if any tweaks were needed)**

```bash
git add -A
git commit -m "chore: redesign verification fixes"
```

---

## Self-Review Notes

- **Spec coverage:** grouped nav (Tasks 5–7), Home/KPIs (Tasks 2, 6), dark theme incl. charts (Tasks 3, 4, 8), accessibility (Tasks 3, 11), deployability via requirements.txt + sys.path shim + graceful no-key (Tasks 7, 9, 10), centralized styling for the author's later pass (Task 3, reused in 5/6/8/9). Future front-end note lives in the spec, not the app — no task needed.
- **Name consistency:** `theme.inject_global_css` / `apply_mpl_style` / `card` / `kpi` and `compute_curve_kpis` are used identically across tasks. The page registry keys ("Yield Curve", "Bond Pricing", "PCA Factors", "Stocks") match between Task 6 and Task 7.
- **No new financial features**; all data/compute reused.
