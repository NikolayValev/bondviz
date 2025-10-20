import streamlit as st
try:
    from streamlit_option_menu import option_menu as _option_menu  # type: ignore
except Exception:
    _option_menu = None
from bondviz.app_logic import compute_pv
from bondviz.visualizer import render_visualizer
from bondviz.stocks_view import render_polygon_stocks
from bondviz.pca_view import render_yield_pca

st.set_page_config(page_title="BondViz", layout="wide")

# Header menu
st.title("BondViz")

def nav_menu() -> str:
    items = ["Treasury Yields", "Curve Calculations", "Stock Picker"]
    if _option_menu is not None:
        with st.container():
            return _option_menu(
                None,
                items,
                icons=["graph-up", "calculator", "bar-chart"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",
            )
    else:
        # Fallback if streamlit-option-menu is not installed
        return st.radio("Navigation", items, horizontal=True)

page = nav_menu()

if page == "Treasury Yields":
    st.header("Treasury Yields")
    render_visualizer()

elif page == "Curve Calculations":
    st.header("Curve Calculations")
    with st.sidebar.form("pv_form"):
        st.subheader("Price a Fixed Coupon Bond")
        F = st.number_input("Face", value=1000.0, step=100.0, key="pv_face")
        c = st.number_input("Coupon rate", value=0.05, step=0.005, format="%.3f", key="pv_coupon")
        y = st.number_input("Continuous yield", value=0.04, step=0.005, format="%.3f", key="pv_yield")
        T = st.number_input("Years to maturity", value=10.0, step=1.0, key="pv_years")
        submitted = st.form_submit_button("Calculate")

    if submitted or "pv_result" not in st.session_state:
        st.session_state["pv_result"] = compute_pv(F, c, y, T)

    st.metric("Present Value", f"{st.session_state['pv_result']:,.2f}")

    st.divider()
    render_yield_pca()

elif page == "Stock Picker":
    st.header("Stock Picker")
    render_polygon_stocks()
