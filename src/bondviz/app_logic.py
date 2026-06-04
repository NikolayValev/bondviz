from __future__ import annotations

import pandas as pd
from datetime import date as Date

from .treasury import latest_par_yields
from .pricing import (
    _bond_cashflows,
    convexity,
    modified_duration,
    pv_continuous,
)


def get_latest_treasury_table() -> tuple[pd.Series, Date, pd.DataFrame]:
    """
    Fetch the latest Treasury par yields and build a simple table for display.

    Returns a tuple of:
      - latest row as a Series
      - the date of the observation (date object)
      - a DataFrame with columns ["Maturity", "Yield (%)"]
    """
    latest = latest_par_yields()
    obs_date = pd.to_datetime(latest["DATE"]).date()

    cols = [c for c in latest.index if str(c).startswith("BC_")]
    table = pd.DataFrame(
        {
            "Maturity": cols,
            "Yield (%)": [latest.get(c, None) for c in cols],
        }
    )
    return latest, obs_date, table


def compute_pv(face_value: float, coupon_rate: float, yield_rate: float, years: float) -> float:
    """Wrapper for present value under continuous compounding."""
    return pv_continuous(face_value, coupon_rate, yield_rate, years)


def compute_bond_metrics(
    face_value: float,
    coupon_rate: float,
    yield_rate: float,
    years: float,
    freq: int = 2,
) -> dict[str, float]:
    """Present value plus modified duration and convexity for a bond.

    Thin wrapper so the UI layer holds no financial math: it builds the shared
    cashflow schedule once and feeds it to the pricing analytics.
    """
    cashflows, times = _bond_cashflows(face_value, coupon_rate, years, freq)
    return {
        "pv": pv_continuous(face_value, coupon_rate, yield_rate, years, freq),
        "modified_duration": modified_duration(cashflows, times, yield_rate),
        "convexity": convexity(cashflows, times, yield_rate),
    }


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

