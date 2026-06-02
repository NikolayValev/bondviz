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
