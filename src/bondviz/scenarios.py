"""Parallel-rate-shift scenario analysis for a fixed-coupon bond.

Pure functions only — they return plain data (a list of dicts) and do no
plotting or UI work, so they're trivially testable and reusable by both
frontends. Everything is continuous compounding, consistent with ``pricing``.
"""
from __future__ import annotations

from typing import List, Mapping, Sequence

from .pricing import (
    _bond_cashflows,
    convexity,
    modified_duration,
    price_from_cashflows,
)


def scenario_shift(
    bond_params: Mapping[str, float], shifts_bps: Sequence[float]
) -> List[dict]:
    """Re-price a bond across a set of parallel yield shifts.

    ``bond_params`` keys: ``face``, ``coupon``, ``yield_``, ``years`` and
    optional ``freq`` (default 2). ``shifts_bps`` is e.g. ``[-100, -50, 0, 50,
    100]``.

    For each shift we return both the full reprice (the accurate number) and
    the duration+convexity approximation of the % change
    (``ΔP/P ≈ −D·Δy + ½·C·(Δy)²``) so callers can show how the quadratic
    approximation tracks the true reprice.
    """
    face = bond_params["face"]
    coupon = bond_params["coupon"]
    base_yield = bond_params["yield_"]
    years = bond_params["years"]
    freq = int(bond_params.get("freq", 2))

    cashflows, times = _bond_cashflows(face, coupon, years, freq)
    base_price = price_from_cashflows(cashflows, times, base_yield)
    dur = modified_duration(cashflows, times, base_yield)
    conv = convexity(cashflows, times, base_yield)

    rows: List[dict] = []
    for shift in shifts_bps:
        dy = shift / 10_000.0
        new_yield = base_yield + dy
        new_price = price_from_cashflows(cashflows, times, new_yield)
        dollar_change = new_price - base_price
        approx_pct = -dur * dy + 0.5 * conv * dy * dy
        rows.append(
            {
                "shift_bps": shift,
                "new_yield": new_yield,
                "new_price": new_price,
                "dollar_change": dollar_change,
                "pct_change": dollar_change / base_price,
                "approx_pct_change": approx_pct,
            }
        )
    return rows
