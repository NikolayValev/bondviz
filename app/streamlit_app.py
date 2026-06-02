import sys
from pathlib import Path

# Make `bondviz` importable when the package isn't pip-installed (e.g. Streamlit Cloud).
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from bondviz import theme  # noqa: E402
from bondviz.home_view import render_home  # noqa: E402
from bondviz.pca_view import render_yield_pca  # noqa: E402
from bondviz.pricing_view import render_pricing  # noqa: E402
from bondviz.stocks_view import render_polygon_stocks  # noqa: E402
from bondviz.visualizer import render_visualizer  # noqa: E402

st.set_page_config(page_title="BondViz", page_icon="📈", layout="wide")
theme.apply_mpl_style()
theme.inject_global_css()

# Pages (functions wrapped as st.Page) + a name->page registry for cross-page links.
yield_curve = st.Page(render_visualizer, title="Yield Curve", icon="📈", url_path="yield-curve")
bond_pricing = st.Page(render_pricing, title="Bond Pricing", icon="🧮", url_path="bond-pricing")
pca_factors = st.Page(render_yield_pca, title="PCA Factors", icon="🧬", url_path="pca")
stocks = st.Page(render_polygon_stocks, title="Stocks", icon="📊", url_path="stocks")

_registry = {
    "Yield Curve": yield_curve,
    "Bond Pricing": bond_pricing,
    "PCA Factors": pca_factors,
    "Stocks": stocks,
}

home = st.Page(
    lambda: render_home(_registry),
    title="Home",
    icon="🏠",
    url_path="home",
    default=True,
)

nav = st.navigation(
    {
        "": [home],
        "Fixed Income": [yield_curve, bond_pricing, pca_factors],
        "Markets": [stocks],
    }
)
nav.run()
