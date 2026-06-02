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
