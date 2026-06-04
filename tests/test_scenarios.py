import pytest

from bondviz.pricing import pv_continuous
from bondviz.pricing_view import _symmetric_shifts
from bondviz.scenarios import scenario_shift

BOND = {"face": 1000.0, "coupon": 0.05, "yield_": 0.04, "years": 10.0, "freq": 2}


def _row(rows, bps):
    return next(r for r in rows if r["shift_bps"] == bps)


def test_zero_shift_returns_base_price_exactly():
    rows = scenario_shift(BOND, [-50, 0, 50])
    base = pv_continuous(BOND["face"], BOND["coupon"], BOND["yield_"], BOND["years"], BOND["freq"])
    zero = _row(rows, 0)
    assert zero["new_price"] == base
    assert zero["dollar_change"] == 0.0
    assert zero["pct_change"] == 0.0
    assert zero["approx_pct_change"] == pytest.approx(0.0)


def test_price_moves_opposite_to_yield():
    rows = scenario_shift(BOND, [-100, 0, 100])
    base = _row(rows, 0)["new_price"]
    assert _row(rows, 100)["new_price"] < base  # yield up -> price down
    assert _row(rows, -100)["new_price"] > base  # yield down -> price up


def test_new_yield_reflects_shift():
    rows = scenario_shift(BOND, [25])
    assert _row(rows, 25)["new_yield"] == pytest.approx(BOND["yield_"] + 0.0025)


def test_approximation_tracks_reprice_for_small_shift():
    rows = scenario_shift(BOND, [25])
    r = _row(rows, 25)
    assert r["approx_pct_change"] == pytest.approx(r["pct_change"], abs=1e-4)


def test_approximation_error_grows_with_shift_size():
    rows = scenario_shift(BOND, [25, 200])
    err_small = abs(_row(rows, 25)["approx_pct_change"] - _row(rows, 25)["pct_change"])
    err_large = abs(_row(rows, 200)["approx_pct_change"] - _row(rows, 200)["pct_change"])
    assert err_small < err_large
    assert err_small < 0.0005  # tiny for ±25 bps


def test_symmetric_shifts_dedups_and_includes_zero():
    assert _symmetric_shifts([100, 25, 50]) == [-100, -50, -25, 0, 25, 50, 100]
    assert _symmetric_shifts([]) == [0]
    assert _symmetric_shifts([50, 50]) == [-50, 0, 50]
