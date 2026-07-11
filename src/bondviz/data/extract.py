"""Network I/O for the Treasury par-yield feed. No parsing business logic beyond
turning one XML <entry> into a plain dict — see transform.py for normalization."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date, datetime
from functools import lru_cache

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from bondviz.treasury import TREASURY_XML, _entry_to_record

_DATE_KEYS = ("DATE", "NEW_DATE", "BC_DATE", "REF_DATE", "RecordDate")


def _record_date(rec: dict) -> date | None:
    for key in _DATE_KEYS:
        raw = rec.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(str(raw)[:10]).date()
        except ValueError:
            continue
    return None


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(requests.RequestException),
)
def _fetch_year_xml(year: int) -> bytes:
    resp = requests.get(TREASURY_XML.format(year=year), timeout=20)
    resp.raise_for_status()
    return resp.content


@lru_cache(maxsize=8)
def _fetch_year_xml_cached(year: int) -> bytes:
    # One network fetch per year per process — backfill loops many dates in the
    # same year, and the feed is one document per year, not per day.
    return _fetch_year_xml(year)


def fetch_curve(target_date: date) -> dict | None:
    """Fetch the raw Treasury payload for one calendar date.

    Returns a JSON-serializable dict holding the verbatim parsed feed record,
    or ``None`` if the feed has no entry for that date (weekend / market
    holiday) — callers must treat that as "no data", not an error.
    """
    xml_bytes = _fetch_year_xml_cached(target_date.year)
    root = ET.fromstring(xml_bytes)
    for entry in root.findall(".//{*}entry"):
        rec = _entry_to_record(entry)
        if _record_date(rec) == target_date:
            return {
                "source": "ust_par_yield",
                "source_url": TREASURY_XML.format(year=target_date.year),
                "target_date": target_date.isoformat(),
                "record": rec,
            }
    return None
