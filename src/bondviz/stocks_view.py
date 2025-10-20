from __future__ import annotations

import os
from datetime import date, timedelta
from typing import List

import pandas as pd
import streamlit as st

from .stocks import fetch_aggregates


def _default_polygon_key() -> str:
    key = os.getenv("POLYGON_API_KEY", "")
    if key:
        return key
    try:
        import streamlit as st  # local import for safety
        return st.secrets["POLYGON_API_KEY"]  # may raise if not configured
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def _cached_aggs(ticker: str, start: date, end: date, multiplier: int, timespan: str, adjusted: bool):
    return fetch_aggregates(ticker, start, end, multiplier=multiplier, timespan=timespan, adjusted=adjusted)


def render_polygon_stocks():
    st.header("Stocks (Polygon.io)")
    with st.form("polygon_form"):
        tickers_csv = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, GOOGL")

        c3, c4, c5, c6 = st.columns(4)
        with c3:
            start = st.date_input("Start", date.today() - timedelta(days=30))
        with c4:
            end = st.date_input("End", date.today())
        with c5:
            timespan = st.selectbox("Timespan", ["day", "hour", "minute"], index=0, help="Minute/hour can return lots of data; use shorter ranges.")
        with c6:
            multiplier = st.number_input("Mult", value=1, min_value=1, step=1)

        adjusted = st.checkbox("Adjusted", value=True)
        submitted = st.form_submit_button("Fetch")

    if not submitted:
        return

    tickers: List[str] = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
    if not tickers:
        st.info("Enter at least one ticker.")
        return

    tabs = st.tabs(tickers)
    for tab, t in zip(tabs, tickers):
        with tab:
            try:
                df = _cached_aggs(t, start, end, int(multiplier), timespan, adjusted)
            except Exception as e:
                st.error(f"Failed to fetch {t}: {e}")
                continue

            if df.empty:
                st.info("No data returned for this range.")
                continue

            st.subheader(f"{t} Price & Volume")
            c1, c2 = st.columns([3, 1])
            with c1:
                st.line_chart(df.set_index("time")["close"], height=280)
            with c2:
                metrics = df.iloc[-1]
                st.metric("Last Close", f"{metrics['close']:.2f}")
                st.metric("Volume", f"{metrics['volume']:,}")

            with st.expander("Raw data"):
                st.dataframe(df, use_container_width=True, hide_index=True)
