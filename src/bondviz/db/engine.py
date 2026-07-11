from functools import lru_cache

from sqlalchemy import Engine, create_engine

from bondviz.config import settings


@lru_cache(maxsize=1)
def get_engine(database_url: str | None = None) -> Engine:
    """Process-wide SQLAlchemy engine, built lazily so importing this module never
    requires DATABASE_URL to be set (e.g. running the Streamlit app without a DB)."""
    url = database_url or settings.database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(url, pool_pre_ping=True, future=True)
