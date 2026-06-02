from __future__ import annotations

import re
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


def _css_rgba_to_mpl(css: str) -> tuple[float, float, float, float]:
    """Convert a CSS 'rgba(r, g, b, a)' string to a normalised (R, G, B, A) tuple
    that matplotlib accepts.  r/g/b are 0-255 integers; a is already 0-1."""
    r, g, b, a = re.match(
        r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)", css
    ).groups()
    return (int(r) / 255, int(g) / 255, int(b) / 255, float(a))


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
        "grid.color": _css_rgba_to_mpl(GRID),
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
