import math
from typing import Iterable, List, Sequence, Tuple

try:
    from . import _native as _native
except Exception:
    _native = None


def _bond_cashflows(
    face_value: float, coupon_rate: float, years: float, freq: int = 2
) -> Tuple[List[float], List[float]]:
    """Discrete coupon schedule for a fixed-coupon bond.

    Returns ``(cashflows, times)`` where ``times`` are payment dates in years
    (``1/freq, 2/freq, …, years``) and ``cashflows`` are the per-period coupons
    (``face * coupon / freq``) with the face value added to the final payment.

    This is the single source of truth for a bond's cashflows — both pricing
    and the duration/convexity analytics build on it so they stay consistent.
    """
    n = max(1, int(round(years * freq)))
    coupon_pmt = face_value * coupon_rate / freq
    times = [(i + 1) / freq for i in range(n)]
    cashflows = [coupon_pmt] * n
    cashflows[-1] += face_value
    return cashflows, times


def price_from_cashflows(
    cashflows: Sequence[float], times: Sequence[float], yield_rate: float
) -> float:
    """Present value of explicit cashflows under continuous compounding."""
    return sum(cf * math.exp(-yield_rate * t) for cf, t in zip(cashflows, times))


def pv_continuous(
    face_value: float, coupon_rate: float, yield_rate: float, years: float, freq: int = 2
) -> float:
    """Present value of a fixed-coupon bond under continuous compounding.

    Discounts a discrete coupon schedule (``freq`` payments per year) so the
    price is consistent with the duration/convexity analytics below.
    """
    if _native is not None:
        try:
            return float(
                _native.pv_continuous(face_value, coupon_rate, yield_rate, years, freq)
            )
        except Exception:
            pass

    cashflows, times = _bond_cashflows(face_value, coupon_rate, years, freq)
    return price_from_cashflows(cashflows, times, yield_rate)


def macaulay_duration(
    cashflows: Sequence[float], times: Sequence[float], yield_: float
) -> float:
    """Macaulay duration = Σ(tᵢ·CFᵢ·e^(−y·tᵢ)) / price."""
    price = price_from_cashflows(cashflows, times, yield_)
    weighted = sum(t * cf * math.exp(-yield_ * t) for cf, t in zip(cashflows, times))
    return weighted / price


def modified_duration(
    cashflows: Sequence[float], times: Sequence[float], yield_: float
) -> float:
    """Modified duration.

    Under continuous compounding modified duration equals Macaulay duration
    (there is no 1/(1+y/m) adjustment), so this delegates to it. It exists as
    its own function because callers expect both — don't "simplify" it away.
    """
    return macaulay_duration(cashflows, times, yield_)


def convexity(
    cashflows: Sequence[float], times: Sequence[float], yield_: float
) -> float:
    """Convexity = Σ(tᵢ²·CFᵢ·e^(−y·tᵢ)) / price."""
    price = price_from_cashflows(cashflows, times, yield_)
    weighted = sum(t * t * cf * math.exp(-yield_ * t) for cf, t in zip(cashflows, times))
    return weighted / price


def discount_factors_continuous(r: float, tenors: Iterable[float]) -> List[Tuple[float, float]]:
    tenor_list = list(tenors)
    if _native is not None:
        try:
            native_result = _native.discount_factors_continuous(r, tenor_list)
            return [(float(t), float(df)) for t, df in native_result]
        except Exception:
            pass
    return [(t, math.exp(-r * t)) for t in tenor_list]
