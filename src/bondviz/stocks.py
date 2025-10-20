from __future__ import annotations

import os
from datetime import date, datetime
from typing import Optional

import pandas as pd


def _load_polygon_client(api_key: Optional[str] = None):
    try:
        from polygon import RESTClient  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "polygon-api-client is not installed. Run `pip install polygon-api-client`"
        ) from e

    # Load .env if present to populate environment variables
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    key = api_key or os.getenv("POLYGON_API_KEY")
    if not key:
        # Streamlit apps can also keep secrets in st.secrets
        try:  # lazy import to avoid hard dependency
            import streamlit as st  # type: ignore

            key = st.secrets.get("POLYGON_API_KEY")  # type: ignore
        except Exception:
            key = None
    if not key:
        raise ValueError(
            "Polygon API key missing. Set POLYGON_API_KEY env var or pass api_key."
        )

    return RESTClient(api_key=key)


def _to_iso(d: date | datetime) -> str:
    if isinstance(d, datetime):
        return d.date().isoformat()
    return d.isoformat()


def fetch_aggregates(
    ticker: str,
    start: date | datetime,
    end: date | datetime,
    *,
    multiplier: int = 1,
    timespan: str = "day",
    adjusted: bool = True,
    sort: str = "asc",
    max_records: int = 50000,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch aggregate bars for a ticker from Polygon.

    Returns a DataFrame with columns: time, open, high, low, close, volume, vwap, transactions
    """
    client = _load_polygon_client(api_key)
    # Use iterator-based API for robust pagination over long ranges
    try:
        iter_aggs = client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=_to_iso(start),
            to=_to_iso(end),
            adjusted=adjusted,
            sort=sort,
            limit=50000,
        )
    except TypeError:
        # Fallback in case list_aggs signature differs; attempt without sort
        iter_aggs = client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=_to_iso(start),
            to=_to_iso(end),
            adjusted=adjusted,
            limit=50000,
        )

    rows = []
    count = 0
    for a in iter_aggs:
        # a may be a dict-like or an object with attributes
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        t_ms = _get(a, "t") or _get(a, "timestamp")
        rows.append(
            {
                "time": pd.to_datetime(t_ms, unit="ms"),
                "open": _get(a, "o") or _get(a, "open"),
                "high": _get(a, "h") or _get(a, "high"),
                "low": _get(a, "l") or _get(a, "low"),
                "close": _get(a, "c") or _get(a, "close"),
                "volume": _get(a, "v") or _get(a, "volume"),
                "vwap": _get(a, "vw") or _get(a, "vwap"),
                "transactions": _get(a, "n") or _get(a, "transactions"),
            }
        )
        count += 1
        if count >= max_records:
            break

    if not rows:
        return pd.DataFrame(columns=["time","open","high","low","close","volume","vwap","transactions"])  # empty

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df


def fetch_previous_close(ticker: str, *, adjusted: bool = True, api_key: Optional[str] = None) -> pd.DataFrame:
    client = _load_polygon_client(api_key)
    resp = client.get_previous_close(ticker, adjusted=adjusted)
    results = getattr(resp, "results", None) or []
    rows = []
    for r in results:
        rows.append(
            {
                "time": pd.to_datetime(r["t"], unit="ms"),
                "open": r.get("o"),
                "high": r.get("h"),
                "low": r.get("l"),
                "close": r.get("c"),
                "volume": r.get("v"),
                "vwap": r.get("vw"),
                "transactions": r.get("n"),
            }
        )
    return pd.DataFrame(rows)


def fetch_snapshot(ticker: str, api_key: Optional[str] = None) -> dict:
    client = _load_polygon_client(api_key)
    snap = client.get_snapshot("stocks", ticker)
    # Convert snapshot to a light dict
    out = {
        "ticker": getattr(snap, "ticker", ticker),
        "day": getattr(snap, "day", None),
        "lastTrade": getattr(snap, "last_trade", None),
        "lastQuote": getattr(snap, "last_quote", None),
        "min": getattr(snap, "min", None),
        "prevDay": getattr(snap, "prev_day", None),
        "todaysChangePct": getattr(snap, "todays_change_percent", None),
    }
    return out
