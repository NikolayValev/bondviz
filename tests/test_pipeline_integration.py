"""Integration test against a real Postgres (a scratch DB / Neon branch — set
DATABASE_URL to run it). Verifies the idempotency guarantee: running the
pipeline twice for the same date must not duplicate or corrupt anything."""

import os
from datetime import date

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="integration test requires DATABASE_URL (a scratch Postgres/Neon branch)",
)

from bondviz.data import pipeline  # noqa: E402
from bondviz.db.engine import get_engine  # noqa: E402


@pytest.fixture
def engine():
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("TRUNCATE etl_run, curve_point, curve, raw_curve_payload RESTART IDENTITY CASCADE")
        )
    yield eng


def _fixed_payload(target_date: date) -> dict:
    return {
        "source": "ust_par_yield",
        "source_url": "https://example.test/fixture",
        "target_date": target_date.isoformat(),
        "record": {
            "NEW_DATE": f"{target_date.isoformat()}T00:00:00",
            "BC_2YEAR": "4.25",
            "BC_10YEAR": "4.00",
        },
    }


def test_ingest_twice_is_idempotent(engine, monkeypatch):
    target = date(2024, 3, 15)
    monkeypatch.setattr(pipeline.extract, "fetch_curve", lambda d: _fixed_payload(target))

    first = pipeline.run(engine, target)
    with engine.connect() as conn:
        curve_count_1 = conn.execute(text("SELECT count(*) FROM curve")).scalar_one()
        point_count_1 = conn.execute(text("SELECT count(*) FROM curve_point")).scalar_one()
        raw_count_1 = conn.execute(text("SELECT count(*) FROM raw_curve_payload")).scalar_one()

    second = pipeline.run(engine, target)
    with engine.connect() as conn:
        curve_count_2 = conn.execute(text("SELECT count(*) FROM curve")).scalar_one()
        point_count_2 = conn.execute(text("SELECT count(*) FROM curve_point")).scalar_one()
        raw_count_2 = conn.execute(text("SELECT count(*) FROM raw_curve_payload")).scalar_one()

    assert first.status == second.status == "success"
    assert curve_count_1 == curve_count_2 == 1
    assert point_count_1 == point_count_2 == 2
    assert raw_count_1 == raw_count_2 == 1  # same payload_hash -> deduped, not re-inserted

    with engine.connect() as conn:
        etl_run_count = conn.execute(text("SELECT count(*) FROM etl_run")).scalar_one()
    assert etl_run_count == 2  # one audit row per invocation, by design — not deduped


def test_no_data_day_is_not_a_failure(engine, monkeypatch):
    target = date(2024, 3, 16)
    monkeypatch.setattr(pipeline.extract, "fetch_curve", lambda d: None)

    result = pipeline.run(engine, target)

    assert result.status == "no_data"
    assert result.rows_ingested == 0
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT status, rows_ingested FROM etl_run WHERE target_date = :d"), {"d": target}
        ).first()
    assert row.status == "no_data"
    assert row.rows_ingested == 0
