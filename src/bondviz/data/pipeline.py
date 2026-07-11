"""Orchestrates one ingest run end to end and logs it to etl_run + stdout.

Every invocation of ``run()`` writes exactly one etl_run row (success, failed,
or no_data) — that audit trail is what turns this from a script into a
pipeline.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

from bondviz.data import extract, load
from bondviz.data.transform import normalize

JOB_NAME = "ingest_curve"


def _log(event: str, **fields: object) -> None:
    line = {"event": event, "ts": datetime.now(timezone.utc).isoformat(), **fields}
    print(json.dumps(line, default=str), file=sys.stdout, flush=True)


@dataclass(frozen=True)
class RunResult:
    status: str  # success | no_data | failed
    rows_ingested: int
    target_date: date
    error_message: str | None = None


def _start_run(conn, target_date: date) -> int:
    row = conn.execute(
        text(
            "INSERT INTO etl_run (job_name, status, target_date) "
            "VALUES (:job_name, 'running', :target_date) RETURNING id"
        ),
        {"job_name": JOB_NAME, "target_date": target_date},
    ).first()
    return row[0]


def _finish_run(
    conn, run_id: int, *, status: str, rows_ingested: int, error_message: str | None
) -> None:
    conn.execute(
        text(
            "UPDATE etl_run SET status = :status, rows_ingested = :rows_ingested, "
            "error_message = :error_message, finished_at = now() WHERE id = :id"
        ),
        {
            "status": status,
            "rows_ingested": rows_ingested,
            "error_message": error_message,
            "id": run_id,
        },
    )


def run(engine: Engine, target_date: date) -> RunResult:
    """Ingest one date: fetch -> raw -> transform -> curated."""
    with engine.begin() as conn:
        run_id = _start_run(conn, target_date)

    _log("ingest.start", target_date=target_date.isoformat(), run_id=run_id)

    try:
        payload = extract.fetch_curve(target_date)
    except Exception as exc:  # feed unreachable after retries, bad XML, etc.
        with engine.begin() as conn:
            _finish_run(conn, run_id, status="failed", rows_ingested=0, error_message=str(exc))
        _log("ingest.failed", target_date=target_date.isoformat(), run_id=run_id, error=str(exc))
        return RunResult("failed", 0, target_date, str(exc))

    if payload is None:
        with engine.begin() as conn:
            _finish_run(conn, run_id, status="no_data", rows_ingested=0, error_message=None)
        _log("ingest.no_data", target_date=target_date.isoformat(), run_id=run_id)
        return RunResult("no_data", 0, target_date)

    try:
        with engine.begin() as conn:
            load.upsert_raw(
                conn, source=payload["source"], source_url=payload["source_url"], payload=payload
            )
            normalized = normalize(payload)
            curve_id = load.upsert_curve(conn, normalized)
            rows = len(normalized.points)
            _finish_run(conn, run_id, status="success", rows_ingested=rows, error_message=None)
    except Exception as exc:  # malformed record, DB constraint violation, etc.
        with engine.begin() as conn:
            _finish_run(conn, run_id, status="failed", rows_ingested=0, error_message=str(exc))
        _log("ingest.failed", target_date=target_date.isoformat(), run_id=run_id, error=str(exc))
        return RunResult("failed", 0, target_date, str(exc))

    _log(
        "ingest.success",
        target_date=target_date.isoformat(),
        run_id=run_id,
        rows_ingested=rows,
        curve_id=curve_id,
    )
    return RunResult("success", rows, target_date)


def backfill(engine: Engine, start: date, end: date, *, sleep_seconds: float = 0.25) -> list[RunResult]:
    """Loop dates from start to end inclusive, skip ones already loaded, rate-limit politely."""
    results: list[RunResult] = []
    current = start
    while current <= end:
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM curve WHERE curve_date = :d LIMIT 1"), {"d": current}
            ).first()
        if exists:
            _log("backfill.skip_existing", target_date=current.isoformat())
        else:
            results.append(run(engine, current))
            time.sleep(sleep_seconds)
        current += timedelta(days=1)
    return results
