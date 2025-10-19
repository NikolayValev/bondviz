import streamlit as st
from bondviz.app_logic import get_latest_treasury_table, compute_pv
from bondviz.visualizer import render_visualizer

st.set_page_config(page_title="BondViz", layout="wide")
st.header("Treasury Yields + Bond PV (Continuous)")

try:
    latest, date, df = get_latest_treasury_table()
    st.subheader(f"Treasury Par Curve — {date}")
    st.dataframe(df, hide_index=True)
except Exception as e:
    st.error(f"Treasury data unavailable: {e}")

st.subheader("Price a Plain-Vanilla Fixed Coupon Bond")
c1, c2, c3, c4 = st.columns(4)
with c1: F = st.number_input("Face", value=1000.0, step=100.0)
with c2: c = st.number_input("Coupon rate", value=0.05, step=0.005, format="%.3f")
with c3: y = st.number_input("Continuous yield", value=0.04, step=0.005, format="%.3f")
with c4: T = st.number_input("Years to maturity", value=10.0, step=1.0)

pv = compute_pv(F, c, y, T)
st.metric("Present Value", f"{pv:,.2f}")

st.divider()
render_visualizer()
