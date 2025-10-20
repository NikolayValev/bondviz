import streamlit as st
from bondviz.app_logic import get_latest_treasury_table, compute_pv
from bondviz.visualizer import render_visualizer
from bondviz.stocks_view import render_polygon_stocks

st.set_page_config(page_title="BondViz", layout="wide")
st.header("Treasury Yields + Bond PV (Continuous)")

try:
    latest, date, df = get_latest_treasury_table()
    st.subheader(f"Treasury Par Curve — {date}")
    st.dataframe(df, hide_index=True)
except Exception as e:
    st.error(f"Treasury data unavailable: {e}")

st.subheader("Price a Plain-Vanilla Fixed Coupon Bond")
with st.form("pv_form"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        F = st.number_input("Face", value=1000.0, step=100.0, key="pv_face")
    with c2:
        c = st.number_input("Coupon rate", value=0.05, step=0.005, format="%.3f", key="pv_coupon")
    with c3:
        y = st.number_input("Continuous yield", value=0.04, step=0.005, format="%.3f", key="pv_yield")
    with c4:
        T = st.number_input("Years to maturity", value=10.0, step=1.0, key="pv_years")
    submitted = st.form_submit_button("Calculate")

if submitted or "pv_result" not in st.session_state:
    st.session_state["pv_result"] = compute_pv(F, c, y, T)

st.metric("Present Value", f"{st.session_state['pv_result']:,.2f}")

st.divider()
render_visualizer()

st.divider()
render_polygon_stocks()
