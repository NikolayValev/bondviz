from __future__ import annotations

import pandas as pd
import streamlit as st

from . import theme
from .app_logic import compute_bond_metrics
from .plots import plot_scenarios
from .scenarios import scenario_shift

_FREQ_LABELS = {"Annual": 1, "Semiannual": 2, "Quarterly": 4}
_SHIFT_MAGNITUDES = [25, 50, 100, 200]


def _symmetric_shifts(magnitudes: list[int]) -> list[int]:
    """Turn selected magnitudes into a sorted symmetric set including 0."""
    shifts = {0}
    for m in magnitudes:
        shifts.add(m)
        shifts.add(-m)
    return sorted(shifts)


def render_pricing() -> None:
    theme.inject_global_css()
    st.header("Bond Pricing")
    st.caption("Present value of a fixed-coupon bond under continuous compounding.")

    with st.sidebar.form("pv_form"):
        st.subheader("Inputs")
        face = st.number_input("Face", value=1000.0, step=100.0, key="pv_face")
        coupon = st.number_input(
            "Coupon rate", value=0.05, step=0.005, format="%.3f", key="pv_coupon"
        )
        ytm = st.number_input(
            "Continuous yield", value=0.04, step=0.005, format="%.3f", key="pv_yield"
        )
        years = st.number_input("Years to maturity", value=10.0, step=1.0, key="pv_years")
        freq_label = st.selectbox(
            "Coupon frequency", list(_FREQ_LABELS), index=1, key="pv_freq"
        )
        submitted = st.form_submit_button("Calculate")

    freq = _FREQ_LABELS[freq_label]

    if submitted or "pv_metrics" not in st.session_state:
        st.session_state["pv_metrics"] = compute_bond_metrics(
            face, coupon, ytm, years, freq
        )

    metrics = st.session_state["pv_metrics"]

    with theme.card():
        st.metric("Present Value", f"{metrics['pv']:,.2f}")
        st.caption(
            f"Face {face:,.0f} · coupon {coupon:.3%} · cont. yield {ytm:.3%}"
            f" · {years:g}y · {freq_label.lower()}"
        )

        c1, c2 = st.columns(2)
        c1.metric("Modified Duration", f"{metrics['modified_duration']:.2f} yrs")
        c2.metric("Convexity", f"{metrics['convexity']:.2f}")

        # First-order + convexity estimate of the price move for a +1% yield shift.
        dy = 0.01
        drop_pct = (
            metrics["modified_duration"] * dy
            - 0.5 * metrics["convexity"] * dy * dy
        ) * 100
        st.caption(f"A 1% rise in yield ≈ {drop_pct:.2f}% drop in price.")

    st.subheader("Scenario Analysis")
    st.caption("Re-price the bond across parallel yield shifts.")

    magnitudes = st.multiselect(
        "Shift magnitudes (± bps)",
        _SHIFT_MAGNITUDES,
        default=[25, 50, 100],
        key="scenario_magnitudes",
    )
    shifts = _symmetric_shifts([int(m) for m in magnitudes])

    bond_params = {
        "face": face,
        "coupon": coupon,
        "yield_": ytm,
        "years": years,
        "freq": freq,
    }
    rows = scenario_shift(bond_params, shifts)

    table = pd.DataFrame(
        {
            "Shift (bps)": [r["shift_bps"] for r in rows],
            "New yield": [f"{r['new_yield']:.3%}" for r in rows],
            "New price": [f"{r['new_price']:,.2f}" for r in rows],
            "$ change": [f"{r['dollar_change']:,.2f}" for r in rows],
            "% change": [f"{r['pct_change']:.2%}" for r in rows],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.pyplot(plot_scenarios(rows))
