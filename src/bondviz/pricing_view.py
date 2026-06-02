from __future__ import annotations

import streamlit as st

from . import theme
from .app_logic import compute_pv


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
        submitted = st.form_submit_button("Calculate")

    if submitted or "pv_result" not in st.session_state:
        st.session_state["pv_result"] = compute_pv(face, coupon, ytm, years)

    with theme.card():
        st.metric("Present Value", f"{st.session_state['pv_result']:,.2f}")
        st.caption(
            f"Face {face:,.0f} · coupon {coupon:.3%} · cont. yield {ytm:.3%} · {years:g}y"
        )
