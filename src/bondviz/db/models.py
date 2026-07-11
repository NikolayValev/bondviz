"""SQLAlchemy Core table metadata mirroring the Alembic-managed schema.

This module is the typed reference for column names/types used when building
queries elsewhere in the data layer; it does not create or alter tables itself
(Alembic migrations under alembic/versions/ are the source of truth for DDL).
"""

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Table,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

raw_curve_payload = Table(
    "raw_curve_payload",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("source", String, nullable=False),
    Column("source_url", String, nullable=False),
    Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("payload", JSONB, nullable=False),
    Column("payload_hash", String, nullable=False),
    UniqueConstraint("source", "payload_hash"),
)

curve = Table(
    "curve",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("curve_date", Date, nullable=False),
    Column("curve_type", String, nullable=False, server_default="par_yield"),
    Column("source", String, nullable=False),
    Column("ingested_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("curve_date", "curve_type", "source"),
)

curve_point = Table(
    "curve_point",
    metadata,
    Column("curve_id", Integer, ForeignKey("curve.id", ondelete="CASCADE"), primary_key=True),
    Column("tenor_months", Integer, primary_key=True),
    Column("yield_pct", Numeric(9, 6), nullable=False),
)

instrument = Table(
    "instrument",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("ticker", String, unique=True),
    Column("cusip", String, unique=True),
    Column("instrument_type", String, nullable=False),
    Column("coupon_rate", Numeric(9, 6), nullable=False),
    Column("face_value", Numeric(18, 2), nullable=False, server_default="100"),
    Column("issue_date", Date, nullable=False),
    Column("maturity_date", Date, nullable=False),
    Column("coupon_freq", SmallInteger, nullable=False, server_default="2"),
    Column("day_count", String, nullable=False, server_default="ACT/ACT"),
    CheckConstraint("maturity_date > issue_date", name="instrument_maturity_after_issue"),
)

pricing_run = Table(
    "pricing_run",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("instrument_id", Integer, ForeignKey("instrument.id"), nullable=False),
    Column("curve_id", Integer, ForeignKey("curve.id"), nullable=False),
    Column("run_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("shift_bps", Integer, nullable=False, server_default="0"),
    Column("clean_price", Numeric(18, 8), nullable=False),
    Column("accrued", Numeric(18, 8), nullable=False),
    Column("ytm", Numeric(9, 6)),
    Column("mac_duration", Numeric(12, 8)),
    Column("mod_duration", Numeric(12, 8)),
    Column("convexity", Numeric(12, 8)),
    # Cost-of-carry columns (added in migration 0002, see bondviz.carry)
    Column("repo_rate", Numeric(9, 6)),
    Column("carry_days", Integer),
    Column("coupon_income", Numeric(18, 8)),
    Column("financing_cost", Numeric(18, 8)),
    Column("net_carry", Numeric(18, 8)),
    UniqueConstraint("instrument_id", "curve_id", "shift_bps"),
)

etl_run = Table(
    "etl_run",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("job_name", String, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("finished_at", DateTime(timezone=True)),
    Column("status", String, nullable=False),
    Column("rows_ingested", Integer, nullable=False, server_default="0"),
    Column("target_date", Date),
    Column("error_message", String),
)
