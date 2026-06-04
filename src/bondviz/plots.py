import matplotlib.pyplot as plt
import pandas as pd
from typing import Mapping, Sequence, Tuple

def plot_yield_curve(latest_row: pd.Series):
    tenors = []
    yields = []
    mapping = [
        ("1M","BC_1MONTH"), ("2M","BC_2MONTH"), ("3M","BC_3MONTH"),
        ("6M","BC_6MONTH"), ("1Y","BC_1YEAR"), ("2Y","BC_2YEAR"),
        ("3Y","BC_3YEAR"), ("5Y","BC_5YEAR"), ("7Y","BC_7YEAR"),
        ("10Y","BC_10YEAR"), ("20Y","BC_20YEAR"), ("30Y","BC_30YEAR")
    ]
    for label, col in mapping:
        if col in latest_row.index and pd.notna(latest_row[col]):
            tenors.append(label)
            yields.append(latest_row[col])
    fig, ax = plt.subplots()
    ax.plot(tenors, yields, marker="o")
    ax.set_title(f"Treasury Par Yield Curve — {pd.to_datetime(latest_row['DATE']).date()}")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Yield (%)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    return fig

def plot_discount_factors(points: Sequence[Tuple[float,float]]):
    t = [p[0] for p in points]
    d = [p[1] for p in points]
    fig, ax = plt.subplots()
    ax.plot(t, d, marker="o")
    ax.set_title("Continuous-Compounding Discount Factors")
    ax.set_xlabel("Years")
    ax.set_ylabel("Discount factor")
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def plot_scenarios(rows: Sequence[Mapping[str, float]]):
    """Bond price vs. parallel yield shift: full reprice vs. the duration+convexity
    approximation, so the convexity gap is visible. ``rows`` is the output of
    ``scenarios.scenario_shift``."""
    shifts = [r["shift_bps"] for r in rows]
    actual = [r["new_price"] for r in rows]
    # Base price is constant across rows: new_price - dollar_change.
    base = rows[0]["new_price"] - rows[0]["dollar_change"]
    approx = [base * (1 + r["approx_pct_change"]) for r in rows]

    fig, ax = plt.subplots()
    ax.plot(shifts, actual, marker="o", label="Actual reprice")
    ax.plot(shifts, approx, marker="s", linestyle="--", label="Duration + convexity")
    ax.set_title("Scenario Analysis — Price vs. Parallel Yield Shift")
    ax.set_xlabel("Yield shift (bps)")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    return fig
