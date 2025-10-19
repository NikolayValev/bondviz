import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd

TREASURY_XML = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}"
)

YIELD_COLS = [
    "BC_1MONTH","BC_2MONTH","BC_3MONTH","BC_6MONTH","BC_1YEAR",
    "BC_2YEAR","BC_3YEAR","BC_5YEAR","BC_7YEAR","BC_10YEAR","BC_20YEAR","BC_30YEAR"
]

def _strip(tag: str) -> str:
    # remove XML namespace: "{ns}TAG" -> "TAG"
    return tag.split("}", 1)[-1] if "}" in tag else tag

def _entry_to_record(entry: ET.Element) -> dict:
    # Prefer nested properties: entry/content/m:properties/*
    props = entry.findall(".//{*}content/{*}properties/*")
    nodes = props if props else list(entry.iter())  # fallback: scan all descendants
    rec = {}
    for el in nodes:
        tag = _strip(el.tag)
        if not tag or tag.lower() in {"content","properties","entry"}:
            continue
        text = (el.text or "").strip()
        if text != "":
            rec[tag] = text
    return rec

def fetch_treasury_par_curve(year: int) -> pd.DataFrame:
    r = requests.get(TREASURY_XML.format(year=year), timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    entries = root.findall(".//{*}entry")
    records = [_entry_to_record(e) for e in entries]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    # Identify date key; Treasury commonly uses NEW_DATE
    date_keys = ["DATE","NEW_DATE","BC_DATE","REF_DATE","RecordDate"]
    date_key = next((k for k in date_keys if k in df.columns), None)
    if not date_key:
        # expose available keys for debugging rather than crashing
        df["DATE"] = pd.NaT
    else:
        df["DATE"] = pd.to_datetime(df[date_key], errors="coerce")

    # Keep only available yield columns
    cols_present = [c for c in YIELD_COLS if c in df.columns]
    keep = ["DATE"] + cols_present
    df = df[keep].copy()

    # Coerce numerics
    for c in cols_present:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    return df

def latest_par_yields() -> pd.Series:
    year = datetime.utcnow().year
    df = fetch_treasury_par_curve(year)
    if df.empty:
        df = fetch_treasury_par_curve(year - 1)
    if df.empty:
        raise RuntimeError("Treasury feed produced no rows")
    return df.iloc[-1]
