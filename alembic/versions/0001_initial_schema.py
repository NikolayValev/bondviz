"""initial schema: raw/curated/operational layers

Revision ID: 0001
Revises:
Create Date: 2026-07-11
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- raw (landing) --------------------------------------------------
    op.execute(
        """
        CREATE TABLE raw_curve_payload (
            id            BIGSERIAL PRIMARY KEY,
            source        TEXT        NOT NULL,
            source_url    TEXT        NOT NULL,
            fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            payload       JSONB       NOT NULL,
            payload_hash  TEXT        NOT NULL,
            UNIQUE (source, payload_hash)
        )
        """
    )

    # --- curated ----------------------------------------------------------
    op.execute(
        """
        CREATE TABLE curve (
            id          SERIAL PRIMARY KEY,
            curve_date  DATE  NOT NULL,
            curve_type  TEXT  NOT NULL DEFAULT 'par_yield',
            source      TEXT  NOT NULL,
            ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (curve_date, curve_type, source)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE curve_point (
            curve_id     INT     NOT NULL REFERENCES curve(id) ON DELETE CASCADE,
            tenor_months INT     NOT NULL,
            yield_pct    NUMERIC(9,6) NOT NULL,
            PRIMARY KEY (curve_id, tenor_months)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE instrument (
            id            SERIAL PRIMARY KEY,
            ticker        TEXT UNIQUE,
            cusip         TEXT UNIQUE,
            instrument_type TEXT NOT NULL,
            coupon_rate   NUMERIC(9,6) NOT NULL,
            face_value    NUMERIC(18,2) NOT NULL DEFAULT 100,
            issue_date    DATE NOT NULL,
            maturity_date DATE NOT NULL,
            coupon_freq   SMALLINT NOT NULL DEFAULT 2,
            day_count     TEXT NOT NULL DEFAULT 'ACT/ACT',
            CHECK (maturity_date > issue_date)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE pricing_run (
            id             BIGSERIAL PRIMARY KEY,
            instrument_id  INT NOT NULL REFERENCES instrument(id),
            curve_id       INT NOT NULL REFERENCES curve(id),
            run_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
            shift_bps      INT NOT NULL DEFAULT 0,
            clean_price    NUMERIC(18,8) NOT NULL,
            accrued        NUMERIC(18,8) NOT NULL,
            ytm            NUMERIC(9,6),
            mac_duration   NUMERIC(12,8),
            mod_duration   NUMERIC(12,8),
            convexity      NUMERIC(12,8),
            UNIQUE (instrument_id, curve_id, shift_bps)
        )
        """
    )

    # --- operational --------------------------------------------------
    op.execute(
        """
        CREATE TABLE etl_run (
            id            BIGSERIAL PRIMARY KEY,
            job_name      TEXT NOT NULL,
            started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at   TIMESTAMPTZ,
            status        TEXT NOT NULL,
            rows_ingested INT  NOT NULL DEFAULT 0,
            target_date   DATE,
            error_message TEXT
        )
        """
    )

    op.execute("CREATE INDEX ix_curve_date ON curve (curve_date DESC)")
    op.execute(
        "CREATE INDEX ix_pricing_run_instrument_run_at ON pricing_run (instrument_id, run_at DESC)"
    )
    op.execute(
        "CREATE INDEX ix_etl_run_job_started_at ON etl_run (job_name, started_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS etl_run")
    op.execute("DROP TABLE IF EXISTS pricing_run")
    op.execute("DROP TABLE IF EXISTS instrument")
    op.execute("DROP TABLE IF EXISTS curve_point")
    op.execute("DROP TABLE IF EXISTS curve")
    op.execute("DROP TABLE IF EXISTS raw_curve_payload")
