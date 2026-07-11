"""Read-side repository layer between the curated schema and the pricing
engine / API. Pricing stays pure (bondviz.pricing takes plain floats); this
module is the only place that knows SQL."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import text
from sqlalchemy.engine import Connection


@dataclass(frozen=True)
class Curve:
    id: int
    curve_date: date
    curve_type: str
    source: str
    points: dict[int, float]  # tenor_months -> yield_pct


def _load_points(conn: Connection, curve_id: int) -> dict[int, float]:
    rows = conn.execute(
        text(
            "SELECT tenor_months, yield_pct FROM curve_point "
            "WHERE curve_id = :id ORDER BY tenor_months"
        ),
        {"id": curve_id},
    ).all()
    return {int(tenor): float(y) for tenor, y in rows}


def _curve_from_row(conn: Connection, row) -> Curve:
    return Curve(
        id=row.id,
        curve_date=row.curve_date,
        curve_type=row.curve_type,
        source=row.source,
        points=_load_points(conn, row.id),
    )


def latest_curve(conn: Connection, *, curve_type: str = "par_yield") -> Curve | None:
    row = conn.execute(
        text(
            "SELECT id, curve_date, curve_type, source FROM curve "
            "WHERE curve_type = :curve_type ORDER BY curve_date DESC LIMIT 1"
        ),
        {"curve_type": curve_type},
    ).first()
    return None if row is None else _curve_from_row(conn, row)


def curve_on(conn: Connection, d: date, *, curve_type: str = "par_yield") -> Curve | None:
    row = conn.execute(
        text(
            "SELECT id, curve_date, curve_type, source FROM curve "
            "WHERE curve_date = :d AND curve_type = :curve_type LIMIT 1"
        ),
        {"d": d, "curve_type": curve_type},
    ).first()
    return None if row is None else _curve_from_row(conn, row)


def curve_history(
    conn: Connection, tenor_months: int, since: date, *, curve_type: str = "par_yield"
) -> list[tuple[date, float]]:
    rows = conn.execute(
        text(
            """
            SELECT c.curve_date, cp.yield_pct
            FROM curve c
            JOIN curve_point cp ON cp.curve_id = c.id
            WHERE cp.tenor_months = :tenor_months
              AND c.curve_type = :curve_type
              AND c.curve_date >= :since
            ORDER BY c.curve_date ASC
            """
        ),
        {"tenor_months": tenor_months, "curve_type": curve_type, "since": since},
    ).all()
    return [(r.curve_date, float(r.yield_pct)) for r in rows]
