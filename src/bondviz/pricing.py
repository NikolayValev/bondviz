import math
from typing import Iterable, List, Tuple

def pv_continuous(face_value: float, coupon_rate: float, yield_rate: float, years: float) -> float:
    C = face_value * coupon_rate
    r = yield_rate
    T = years
    if r == 0.0:
        return C * T + face_value
    return C * (1 - math.exp(-r * T)) / r + face_value * math.exp(-r * T)

def discount_factors_continuous(r: float, tenors: Iterable[float]) -> List[Tuple[float, float]]:
    return [(t, math.exp(-r * t)) for t in tenors]
