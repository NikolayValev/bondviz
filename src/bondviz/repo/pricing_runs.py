"""Write-side repository for persisting pricing results. Pricing math itself
stays in bondviz.pricing / bondviz.carry — this module only knows SQL."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection


def record_pricing_run(
    conn: Connection,
    *,
    instrument_id: int,
    curve_id: int,
    shift_bps: int,
    clean_price: float,
    accrued: float,
    ytm: float | None = None,
    mac_duration: float | None = None,
    mod_duration: float | None = None,
    convexity: float | None = None,
    repo_rate: float | None = None,
    carry_days: int | None = None,
    coupon_income: float | None = None,
    financing_cost: float | None = None,
    net_carry: float | None = None,
) -> int:
    """Idempotent upsert keyed on (instrument_id, curve_id, shift_bps) — re-running
    the same scenario overwrites its row rather than duplicating it."""
    result = conn.execute(
        text(
            """
            INSERT INTO pricing_run (
                instrument_id, curve_id, shift_bps, clean_price, accrued,
                ytm, mac_duration, mod_duration, convexity,
                repo_rate, carry_days, coupon_income, financing_cost, net_carry
            ) VALUES (
                :instrument_id, :curve_id, :shift_bps, :clean_price, :accrued,
                :ytm, :mac_duration, :mod_duration, :convexity,
                :repo_rate, :carry_days, :coupon_income, :financing_cost, :net_carry
            )
            ON CONFLICT (instrument_id, curve_id, shift_bps) DO UPDATE SET
                run_at = now(),
                clean_price = EXCLUDED.clean_price,
                accrued = EXCLUDED.accrued,
                ytm = EXCLUDED.ytm,
                mac_duration = EXCLUDED.mac_duration,
                mod_duration = EXCLUDED.mod_duration,
                convexity = EXCLUDED.convexity,
                repo_rate = EXCLUDED.repo_rate,
                carry_days = EXCLUDED.carry_days,
                coupon_income = EXCLUDED.coupon_income,
                financing_cost = EXCLUDED.financing_cost,
                net_carry = EXCLUDED.net_carry
            RETURNING id
            """
        ),
        {
            "instrument_id": instrument_id,
            "curve_id": curve_id,
            "shift_bps": shift_bps,
            "clean_price": clean_price,
            "accrued": accrued,
            "ytm": ytm,
            "mac_duration": mac_duration,
            "mod_duration": mod_duration,
            "convexity": convexity,
            "repo_rate": repo_rate,
            "carry_days": carry_days,
            "coupon_income": coupon_income,
            "financing_cost": financing_cost,
            "net_carry": net_carry,
        },
    )
    return result.scalar_one()
