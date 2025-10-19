from __future__ import annotations

import pandas as pd
from datetime import date as Date

from .treasury import latest_par_yields
from .pricing import pv_continuous


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

