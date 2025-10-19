import matplotlib.pyplot as plt
import pandas as pd
from typing import Sequence, Tuple

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
