# BondViz

A Python data platform for U.S. Treasury yield curves: a scheduled ETL pipeline lands
daily par-yield data in Postgres, a repository layer serves it to a bond-pricing engine
(continuous compounding, duration/convexity, cost-of-carry), and a Streamlit dashboard
visualizes it.

---

## Overview

BondViz provides:

- A scheduled ETL pipeline ingesting the daily U.S. Treasury par-yield curve into
  Postgres — raw → curated layering, idempotent upserts, backfill, run-level audit log
- A repository layer + minimal FastAPI (`GET /curves/*`, `POST /price`) serving stored
  curves to the pricing engine and persisting results to `pricing_run`
- Continuous-compounding bond valuation, duration/convexity, and cost-of-carry / repo
  financing math
- Matplotlib visualizations of yield curves and discount factors
- Streamlit dashboard for interactive exploration
- Polygon.io stock data integration
- Yield-curve PCA explorer (scikit-learn)
- Optional pybind11 C++ extension for fast bond math

---

## Architecture

```
U.S. Treasury XML feed
        │  extract.py (fetch, retry w/ tenacity)
        ▼
raw_curve_payload  ── raw layer ──  immutable, JSONB, dedup'd by payload hash
        │  transform.py (pure: dict -> CurvePoint[])
        ▼
curve / curve_point ── curated layer ──  one row per (date, tenor), upserted
        │  bondviz/repo/curves.py (SQL, no business logic)
        ▼
bondviz/pricing.py + bondviz/carry.py  ── pure math, no DB ──
        │
        ├─ bondviz/api/main.py  (FastAPI: GET /curves/*, POST /price) ─┐
        │                                                              ▼
        └─ app/streamlit_app.py  (Streamlit dashboard)          pricing_run (audit)

Every ETL run (success, failure, or no-data day) writes one row to etl_run.
```

**Why layered this way:** the raw table is immutable and dumb — if a transform bug
surfaces later, history replays from `raw_curve_payload` without re-hitting the Treasury
feed. `transform.py` is pure (no I/O), so it's unit-testable against saved fixtures
without a database. The curated tables (`curve`, `curve_point`) are what everything else
reads, upserted idempotently so re-running a day's job never duplicates data. Pricing
(`bondviz/pricing.py`, `bondviz/carry.py`) stays pure and DB-free; only the repository
layer (`bondviz/repo/`) and the API know SQL. `etl_run` makes every job execution an
auditable event instead of a fire-and-forget script.

### Schema (Postgres, managed by Alembic — see `alembic/versions/`)

| Table | Layer | Purpose |
| --- | --- | --- |
| `raw_curve_payload` | Raw | Verbatim parsed feed response per fetch, deduped by `(source, payload_hash)` |
| `curve` | Curated | One row per `(curve_date, curve_type, source)` |
| `curve_point` | Curated | `(curve_id, tenor_months)` → `yield_pct` |
| `instrument` | Curated | Priceable bonds (ticker, coupon, maturity, day count) |
| `pricing_run` | Curated | One pricing scenario per `(instrument_id, curve_id, shift_bps)`, incl. carry columns |
| `etl_run` | Operational | Audit log: one row per ingest invocation (`running`/`success`/`failed`/`no_data`) |

### Data layer package layout

```
src/bondviz/
  data/
    extract.py    # fetch_curve(date) -> raw payload dict, tenacity retries
    transform.py  # normalize(payload) -> NormalizedCurve  (pure, no I/O)
    load.py       # upsert_raw(), upsert_curve()  (idempotent)
    pipeline.py   # run(date) / backfill(start, end) — orchestrates + logs etl_run
  db/
    engine.py     # SQLAlchemy engine from DATABASE_URL
    models.py     # Core table metadata (reference; Alembic owns the DDL)
  repo/
    curves.py     # latest_curve(), curve_on(), curve_history()  (read SQL)
    pricing_runs.py  # record_pricing_run()  (write SQL)
  carry.py        # cost-of-carry / repo financing math (pure)
  api/main.py     # FastAPI: GET /curves/latest|{date}|history, POST /price
  cli.py          # `python -m bondviz.cli ingest|backfill`
alembic/versions/ # 0001 initial schema, 0002 pricing_run carry columns
```

---

## Project Structure

bondviz/
├── pyproject.toml
├── alembic.ini, alembic/ # DB migrations (see Architecture above)
├── src/bondviz/
│ ├── init.py
│ ├── pricing.py # Continuous compounding logic
│ ├── carry.py # Cost-of-carry / repo financing
│ ├── treasury.py # Fetch & parse Treasury XML (Streamlit path)
│ ├── plots.py # Matplotlib visualization
│ ├── data/, db/, repo/, api/, cli.py # ETL + Postgres data layer
├── scripts/
│ ├── fetch_treasury.py # CLI: fetch yield data
│ ├── price_bond.py # CLI: compute PV
│ ├── win/ # Windows startup scripts
│ └── unix/ # Unix/WSL startup scripts
├── .github/workflows/ # ingest-curve.yml (cron), ci.yml, keepalive.yml
├── Dockerfile
└── app/
└── streamlit_app.py # Web dashboard

---

## Setup

### 1. Create and install dependencies

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.venv\Scripts\python.exe -m pip install -e .
```

#### Alternatively use helper script

```bash
scripts\win\setup_venv.bat
```

### 2. Native extension (optional)

The project ships with a pybind11-backed extension (`bondviz._native`) that accelerates bond pricing.

- Windows: install Microsoft Build Tools (C++ workload) before running the setup script.  
- macOS/Linux: ensure clang/gcc and headers are available (`xcode-select --install` or `build-essential`).  
- Run `pip install -e .` (or the helper script) to compile automatically. If compilation fails, the app falls back to Python implementations.

### Usage

Command-line tools

Fetch latest yield curve:

```bash
scripts\win\run_cli_fetch.bat
```

Price a bond:

```bash
scripts\win\run_cli_price.bat
```

## Streamlit App

```bash
scripts\win\run_streamlit.bat
```

or

```bash
python -m streamlit run app\streamlit_app.py
```

Open the local URL (default <http://localhost:8501>) to view:

Current Treasury yield curve

Continuous-compounding bond PV calculator

## Screenshots



## Data Layer (ETL + Postgres)

### 1. Install the extras

```bash
.venv/bin/pip install -e ".[etl,api]"
```

### 2. Point at a Postgres database

Copy `.env.example` to `.env` and fill in `DATABASE_URL` (a
[Neon](https://neon.tech) free-tier project works well — scale-to-zero, no card).

```bash
cp .env.example .env
```

### 3. Run migrations

```bash
alembic upgrade head
```

### 4. Ingest data

```bash
# One date
python -m bondviz.cli ingest --date 2026-07-10

# A range (skips dates already loaded, rate-limits requests)
python -m bondviz.cli backfill --from 2024-01-01 --to 2026-07-10
```

Each invocation writes one row to `etl_run` (`success` / `failed` / `no_data` — Treasury
publishes nothing on weekends and market holidays, which is `no_data`, not a failure).

### 5. Serve the API

```bash
uvicorn bondviz.api.main:app --reload
```

- `GET /curves/latest`, `GET /curves/{date}`
- `GET /curves/history?tenor=10y&since=2024-01-01`
- `POST /price` — prices a bond against a stored curve and persists the result
  (incl. cost-of-carry, if `repo_rate`/`carry_days` are supplied) to `pricing_run`

### 6. Scheduled ingestion (GitHub Actions)

`.github/workflows/ingest-curve.yml` runs on a weekday cron (22:30 UTC, after the US
close) and on `workflow_dispatch` (for manual runs and backfills). Set the `DATABASE_URL`
repo secret before enabling it. `.github/workflows/keepalive.yml` pushes a monthly no-op
commit so GitHub doesn't auto-disable the cron after 60 days of repo inactivity.

### 7. Docker

```bash
docker build -t bondviz .
docker run --rm -p 8000:8000 -e DATABASE_URL=... bondviz
# or run the CLI instead of the API:
docker run --rm -e DATABASE_URL=... bondviz python -m bondviz.cli ingest --date 2026-07-10
```

## Notes

Uses official Treasury XML feed (no API key required)

Handles namespace changes and missing date tags gracefully

Compatible with Windows, WSL, and macOS

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `app/streamlit_app.py`.
3. Streamlit installs `requirements.txt` (pure Python — no C++ build; bond math falls
   back to the Python implementation when the native extension is absent).
4. Home and Fixed Income work with no configuration. For the Stocks page, add a secret
   under **Manage app → Settings → Secrets**:

   ```toml
   POLYGON_API_KEY = "your_key_here"
   ```

## Web front-end (Vercel)

A native Next.js + D3 front-end lives in [`web/`](web/) and deploys to Vercel at a custom domain
(see [`web/README.md`](web/README.md)). It reuses the same Treasury data and is independent of the
Streamlit app.

### License

MIT License
