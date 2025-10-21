import math
from typing import Iterable, List, Tuple

try:
    from . import _native as _native
except Exception:
    _native = None

def pv_continuous(face_value: float, coupon_rate: float, yield_rate: float, years: float) -> float:
    if _native is not None:
        try:
            return float(_native.pv_continuous(face_value, coupon_rate, yield_rate, years))
        except Exception:
            pass

    C = face_value * coupon_rate
    r = yield_rate
    T = years
    if r == 0.0:
        return C * T + face_value
    return C * (1 - math.exp(-r * T)) / r + face_value * math.exp(-r * T)

def discount_factors_continuous(r: float, tenors: Iterable[float]) -> List[Tuple[float, float]]:
    tenor_list = list(tenors)
    if _native is not None:
        try:
            native_result = _native.discount_factors_continuous(r, tenor_list)
            return [(float(t), float(df)) for t, df in native_result]
        except Exception:
            pass
    return [(t, math.exp(-r * t)) for t in tenor_list]
