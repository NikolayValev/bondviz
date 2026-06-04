import math

import pytest

from bondviz.pricing import (
    _bond_cashflows,
    convexity,
    macaulay_duration,
    modified_duration,
    pv_continuous,
)


def test_pv_continuous_is_discrete_cashflow_sum():
    # pv_continuous now sums discrete coupon cashflows, each discounted with
    # continuous compounding: PV = sum(CF_i * exp(-y * t_i)).
    face, coupon, y, years, freq = 1000.0, 0.05, 0.04, 10.0, 2
    cashflows, times = _bond_cashflows(face, coupon, years, freq)
    expected = sum(cf * math.exp(-y * t) for cf, t in zip(cashflows, times))
    assert pv_continuous(face, coupon, y, years) == pytest.approx(expected)


def test_bond_cashflows_structure():
    cashflows, times = _bond_cashflows(1000.0, 0.05, 10.0, 2)
    assert len(cashflows) == len(times) == 20
    assert times[0] == pytest.approx(0.5)
    assert times[-1] == pytest.approx(10.0)
    # Each coupon = face * coupon / freq; final adds face.
    assert cashflows[0] == pytest.approx(25.0)
    assert cashflows[-1] == pytest.approx(1025.0)


def test_zero_coupon_macaulay_duration_equals_maturity():
    cashflows, times = _bond_cashflows(1000.0, 0.0, 7.0, 2)
    assert macaulay_duration(cashflows, times, 0.04) == pytest.approx(7.0)


def test_modified_equals_macaulay_under_continuous_compounding():
    cashflows, times = _bond_cashflows(1000.0, 0.05, 10.0, 2)
    y = 0.04
    assert modified_duration(cashflows, times, y) == pytest.approx(
        macaulay_duration(cashflows, times, y)
    )


def test_duration_and_convexity_positive_for_coupon_bond():
    cashflows, times = _bond_cashflows(1000.0, 0.05, 10.0, 2)
    y = 0.04
    assert macaulay_duration(cashflows, times, y) > 0
    assert convexity(cashflows, times, y) > 0


def test_longer_maturity_has_larger_duration():
    y = 0.04
    short = _bond_cashflows(1000.0, 0.05, 5.0, 2)
    long = _bond_cashflows(1000.0, 0.05, 20.0, 2)
    assert macaulay_duration(*long, y) > macaulay_duration(*short, y)


def test_duration_predicts_small_yield_bump():
    face, coupon, y, years, freq = 1000.0, 0.05, 0.04, 10.0, 2
    cashflows, times = _bond_cashflows(face, coupon, years, freq)
    base = pv_continuous(face, coupon, y, years)
    bump = 0.0010  # +10 bps
    bumped = pv_continuous(face, coupon, y + bump, years)

    d = modified_duration(cashflows, times, y)
    predicted = base * (1 - d * bump)
    # First-order prediction: right sign and rough magnitude (loose tol).
    assert bumped < base
    assert predicted == pytest.approx(bumped, rel=1e-3)
