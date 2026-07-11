"""``python -m bondviz.cli`` — ETL entry point.

    python -m bondviz.cli ingest --date 2026-07-10
    python -m bondviz.cli backfill --from 2024-01-01 --to 2026-07-10
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from bondviz.data import pipeline
from bondviz.db.engine import get_engine


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _cmd_ingest(args: argparse.Namespace) -> int:
    target_date = _parse_date(args.date) if args.date else date.today()
    engine = get_engine()
    result = pipeline.run(engine, target_date)
    return 1 if result.status == "failed" else 0


def _cmd_backfill(args: argparse.Namespace) -> int:
    start = _parse_date(args.date_from)
    end = _parse_date(args.date_to)
    if end < start:
        print("--to must not be before --from", file=sys.stderr)
        return 2
    engine = get_engine()
    results = pipeline.backfill(engine, start, end, sleep_seconds=args.sleep_seconds)
    return 1 if any(r.status == "failed" for r in results) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bondviz.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest one date's par yield curve")
    ingest_p.add_argument("--date", help="YYYY-MM-DD, defaults to today (UTC)")
    ingest_p.set_defaults(func=_cmd_ingest)

    backfill_p = sub.add_parser("backfill", help="Ingest a range of dates, skipping ones already present")
    backfill_p.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    backfill_p.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    backfill_p.add_argument(
        "--sleep-seconds", type=float, default=0.25, help="pause between requests (politeness)"
    )
    backfill_p.set_defaults(func=_cmd_backfill)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
