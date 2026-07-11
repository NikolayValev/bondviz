"""Pure transform: raw payload dict -> normalized curve records.

No network, no database — this is what makes the pipeline unit-testable
without a DB, and it's the layer fixtures in tests/fixtures/ exercise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

# Mirrors visualizer.py's BC_TO_TENOR, expressed in months (the curated grain).
BC_TO_TENOR_MONTHS: dict[str, int] = {
    "BC_1MONTH": 1,
    "BC_2MONTH": 2,
    "BC_3MONTH": 3,
    "BC_6MONTH": 6,
    "BC_1YEAR": 12,
    "BC_2YEAR": 24,
    "BC_3YEAR": 36,
    "BC_5YEAR": 60,
    "BC_7YEAR": 84,
    "BC_10YEAR": 120,
    "BC_20YEAR": 240,
    "BC_30YEAR": 360,
}

_DATE_KEYS = ("DATE", "NEW_DATE", "BC_DATE", "REF_DATE", "RecordDate")


class TransformError(ValueError):
    """Raised when a raw payload cannot be normalized into curve points."""


@dataclass(frozen=True)
class CurvePoint:
    tenor_months: int
    yield_pct: float


@dataclass(frozen=True)
class NormalizedCurve:
    curve_date: date
    curve_type: str
    source: str
    points: list[CurvePoint]


def _parse_date(record: dict) -> date:
    for key in _DATE_KEYS:
        raw = record.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(str(raw)[:10]).date()
        except ValueError:
            continue
    raise TransformError(f"no parseable date field in record; keys={sorted(record)}")


def normalize(payload: dict) -> NormalizedCurve:
    """Normalize one fetched payload into a curve header + its tenor points.

    Raises TransformError for malformed input (missing/empty record, no
    parseable date, no recognized tenor columns) rather than crashing.
    """
    record = payload.get("record")
    if not isinstance(record, dict) or not record:
        raise TransformError("payload missing a non-empty 'record' dict")

    curve_date = _parse_date(record)

    points: list[CurvePoint] = []
    for bc_key, tenor_months in BC_TO_TENOR_MONTHS.items():
        raw_value = record.get(bc_key)
        if raw_value in (None, ""):
            continue
        try:
            yield_pct = float(raw_value)
        except (TypeError, ValueError):
            continue
        points.append(CurvePoint(tenor_months=tenor_months, yield_pct=yield_pct))

    if not points:
        raise TransformError(f"no recognized tenor columns for {curve_date.isoformat()}")

    return NormalizedCurve(
        curve_date=curve_date,
        curve_type="par_yield",
        source=payload.get("source", "ust_par_yield"),
        points=sorted(points, key=lambda p: p.tenor_months),
    )
