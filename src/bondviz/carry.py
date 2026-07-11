"""Cost-of-carry / repo financing math for a long bond position.

Pure functions, no I/O — mirrors the pricing.py convention so this stays
unit-testable and composable with the repository layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Carry:
    coupon_income: float
    financing_cost: float
    net_carry: float
    breakeven_repo_rate: float


def carry(
    *,
    dirty_price: float,
    coupon_rate: float,
    face_value: float,
    repo_rate: float,
    days: int,
    day_count: str = "ACT/360",
) -> Carry:
    """Financing a long bond position in repo.

    coupon_income  = accrued coupon over the holding period
    financing_cost = dirty_price * repo_rate * days / basis
    net_carry      = coupon_income - financing_cost

    Positive net carry = the coupon out-earns the financing (positive carry).
    ``breakeven_repo_rate`` is the repo rate at which net_carry flips to zero.
    """
    basis = 360 if day_count.upper() == "ACT/360" else 365
    coupon_income = face_value * coupon_rate * days / basis
    financing_cost = dirty_price * repo_rate * days / basis
    net_carry = coupon_income - financing_cost
    breakeven_repo_rate = (
        (coupon_income * basis) / (dirty_price * days) if dirty_price and days else 0.0
    )
    return Carry(
        coupon_income=coupon_income,
        financing_cost=financing_cost,
        net_carry=net_carry,
        breakeven_repo_rate=breakeven_repo_rate,
    )
