"""Minimal read/price API over the curated schema.

Kept separate from the Streamlit app and the Next.js frontend — this is the
interface a future consumer (or `web/`) calls to get curve data or price a
bond against a stored curve. Pricing math itself lives in bondviz.pricing /
bondviz.carry (pure); this module only wires HTTP <-> repo <-> pricing.
"""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from bondviz.carry import carry as compute_carry
from bondviz.db.engine import get_engine
from bondviz.pricing import (
    _bond_cashflows,
    convexity as compute_convexity,
    macaulay_duration,
    modified_duration,
    price_from_cashflows,
)
from bondviz.repo.curves import Curve, curve_history, curve_on, latest_curve
from bondviz.repo.pricing_runs import record_pricing_run

app = FastAPI(title="BondViz Data API")

TENOR_LABEL_TO_MONTHS = {
    "1m": 1, "2m": 2, "3m": 3, "6m": 6,
    "1y": 12, "2y": 24, "3y": 36, "5y": 60,
    "7y": 84, "10y": 120, "20y": 240, "30y": 360,
}


def _curve_out(c: Curve) -> dict:
    return {
        "id": c.id,
        "curve_date": c.curve_date.isoformat(),
        "curve_type": c.curve_type,
        "source": c.source,
        "points": [{"tenor_months": t, "yield_pct": y} for t, y in sorted(c.points.items())],
    }


@app.get("/curves/latest")
def get_latest_curve() -> dict:
    with get_engine().connect() as conn:
        c = latest_curve(conn)
    if c is None:
        raise HTTPException(status_code=404, detail="no curve data ingested yet")
    return _curve_out(c)


@app.get("/curves/history")
def get_curve_history(
    tenor: str = Query(..., description="e.g. '10y', '6m'"),
    since: date = Query(...),
) -> dict:
    # Registered before /curves/{curve_date} — Starlette matches routes in
    # registration order, and "history" would otherwise fail to parse as a date.
    months = TENOR_LABEL_TO_MONTHS.get(tenor.lower())
    if months is None:
        raise HTTPException(status_code=400, detail=f"unrecognized tenor '{tenor}'")
    with get_engine().connect() as conn:
        rows = curve_history(conn, months, since)
    return {
        "tenor": tenor.lower(),
        "tenor_months": months,
        "points": [{"curve_date": d.isoformat(), "yield_pct": y} for d, y in rows],
    }


@app.get("/curves/{curve_date}")
def get_curve_on(curve_date: date) -> dict:
    with get_engine().connect() as conn:
        c = curve_on(conn, curve_date)
    if c is None:
        raise HTTPException(status_code=404, detail=f"no curve for {curve_date.isoformat()}")
    return _curve_out(c)


class PriceRequest(BaseModel):
    instrument_id: int
    face_value: float
    coupon_rate: float
    years: float
    freq: int = 2
    shift_bps: int = 0
    curve_date: date | None = None  # defaults to the latest ingested curve
    tenor: str = "10y"  # which curve point to use as the discount yield
    repo_rate: float | None = None
    carry_days: int | None = None


@app.post("/price")
def price_bond(req: PriceRequest) -> dict:
    months = TENOR_LABEL_TO_MONTHS.get(req.tenor.lower())
    if months is None:
        raise HTTPException(status_code=400, detail=f"unrecognized tenor '{req.tenor}'")

    engine = get_engine()
    with engine.connect() as conn:
        c = curve_on(conn, req.curve_date) if req.curve_date else latest_curve(conn)
    if c is None:
        raise HTTPException(status_code=404, detail="no curve data available")
    if months not in c.points:
        raise HTTPException(status_code=404, detail=f"curve has no {req.tenor} point")

    yield_rate = (c.points[months] + req.shift_bps / 100.0) / 100.0

    cashflows, times = _bond_cashflows(req.face_value, req.coupon_rate, req.years, req.freq)
    clean_price = price_from_cashflows(cashflows, times, yield_rate)
    mac_dur = macaulay_duration(cashflows, times, yield_rate)
    mod_dur = modified_duration(cashflows, times, yield_rate)
    convexity_val = compute_convexity(cashflows, times, yield_rate)
    accrued = 0.0  # no settlement-date accrual model yet — clean price is the full PV

    carry_out = None
    if req.repo_rate is not None and req.carry_days is not None:
        carry_out = compute_carry(
            dirty_price=clean_price + accrued,
            coupon_rate=req.coupon_rate,
            face_value=req.face_value,
            repo_rate=req.repo_rate,
            days=req.carry_days,
        )

    with engine.begin() as conn:
        run_id = record_pricing_run(
            conn,
            instrument_id=req.instrument_id,
            curve_id=c.id,
            shift_bps=req.shift_bps,
            clean_price=clean_price,
            accrued=accrued,
            ytm=yield_rate * 100,
            mac_duration=mac_dur,
            mod_duration=mod_dur,
            convexity=convexity_val,
            repo_rate=req.repo_rate,
            carry_days=req.carry_days,
            coupon_income=carry_out.coupon_income if carry_out else None,
            financing_cost=carry_out.financing_cost if carry_out else None,
            net_carry=carry_out.net_carry if carry_out else None,
        )

    return {
        "pricing_run_id": run_id,
        "curve_id": c.id,
        "curve_date": c.curve_date.isoformat(),
        "clean_price": clean_price,
        "accrued": accrued,
        "ytm_pct": yield_rate * 100,
        "mac_duration": mac_dur,
        "mod_duration": mod_dur,
        "convexity": convexity_val,
        "carry": None
        if carry_out is None
        else {
            "coupon_income": carry_out.coupon_income,
            "financing_cost": carry_out.financing_cost,
            "net_carry": carry_out.net_carry,
            "breakeven_repo_rate_pct": carry_out.breakeven_repo_rate * 100,
        },
    }
