FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic

RUN pip install --no-cache-dir -e ".[etl,api]"

ENV DATABASE_URL=""

# Default: serve the read/price API. Override the command to run the ETL CLI
# instead, e.g.:
#   docker run --rm -e DATABASE_URL=... bondviz python -m bondviz.cli ingest --date 2026-07-10
CMD ["uvicorn", "bondviz.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
