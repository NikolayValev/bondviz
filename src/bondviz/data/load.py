"""Idempotent writes into the raw and curated layers. All functions take an
open SQLAlchemy Connection/transaction — callers control commit boundaries."""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import text
from sqlalchemy.engine import Connection

from bondviz.data.transform import NormalizedCurve


def payload_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def upsert_raw(conn: Connection, *, source: str, source_url: str, payload: dict) -> int | None:
    """Insert the verbatim payload into the raw landing table.

    Returns the new row id, or ``None`` when this exact payload was already
    stored — dedupe on ``(source, payload_hash)`` is the idempotency
    guarantee for the raw layer.
    """
    result = conn.execute(
        text(
            """
            INSERT INTO raw_curve_payload (source, source_url, payload, payload_hash)
            VALUES (:source, :source_url, CAST(:payload AS JSONB), :payload_hash)
            ON CONFLICT (source, payload_hash) DO NOTHING
            RETURNING id
            """
        ),
        {
            "source": source,
            "source_url": source_url,
            "payload": json.dumps(payload, default=str),
            "payload_hash": payload_hash(payload),
        },
    )
    row = result.first()
    return row[0] if row else None


def upsert_curve(conn: Connection, curve: NormalizedCurve) -> int:
    """Upsert the curve header and replace its points.

    Idempotent: re-running for the same ``(curve_date, curve_type, source)``
    reuses the same curve row and leaves ``curve_point`` in the same final
    state — never duplicates or corrupts it.
    """
    result = conn.execute(
        text(
            """
            INSERT INTO curve (curve_date, curve_type, source)
            VALUES (:curve_date, :curve_type, :source)
            ON CONFLICT (curve_date, curve_type, source)
            DO UPDATE SET ingested_at = now()
            RETURNING id
            """
        ),
        {"curve_date": curve.curve_date, "curve_type": curve.curve_type, "source": curve.source},
    )
    curve_id = result.scalar_one()

    conn.execute(text("DELETE FROM curve_point WHERE curve_id = :curve_id"), {"curve_id": curve_id})
    conn.execute(
        text(
            "INSERT INTO curve_point (curve_id, tenor_months, yield_pct) "
            "VALUES (:curve_id, :tenor_months, :yield_pct)"
        ),
        [
            {"curve_id": curve_id, "tenor_months": p.tenor_months, "yield_pct": p.yield_pct}
            for p in curve.points
        ],
    )
    return curve_id
