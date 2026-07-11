import pytest

from bondviz.carry import carry


def test_positive_carry_when_coupon_beats_repo():
    result = carry(dirty_price=1000.0, coupon_rate=0.05, face_value=1000.0, repo_rate=0.03, days=30)
    assert result.coupon_income > result.financing_cost
    assert result.net_carry > 0


def test_negative_carry_when_repo_beats_coupon():
    result = carry(dirty_price=1000.0, coupon_rate=0.02, face_value=1000.0, repo_rate=0.05, days=30)
    assert result.net_carry < 0


def test_breakeven_repo_rate_zeroes_net_carry():
    result = carry(dirty_price=1010.0, coupon_rate=0.045, face_value=1000.0, repo_rate=0.03, days=45)
    at_breakeven = carry(
        dirty_price=1010.0,
        coupon_rate=0.045,
        face_value=1000.0,
        repo_rate=result.breakeven_repo_rate,
        days=45,
    )
    assert at_breakeven.net_carry == pytest.approx(0.0, abs=1e-9)


def test_act_360_accrues_more_than_act_365():
    r360 = carry(
        dirty_price=1000, coupon_rate=0.05, face_value=1000, repo_rate=0.03, days=30, day_count="ACT/360"
    )
    r365 = carry(
        dirty_price=1000, coupon_rate=0.05, face_value=1000, repo_rate=0.03, days=30, day_count="ACT/365"
    )
    assert r360.coupon_income > r365.coupon_income
