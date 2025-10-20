from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date as Date, timedelta, date

try:
    # Prefer local bondviz sources if available
    from .treasury import latest_par_yields, fetch_treasury_par_curve  # type: ignore
    USE_BONDVIZ = True
except Exception as e:  # pragma: no cover - optional fallback path
    USE_BONDVIZ = False
    latest_par_yields = None  # type: ignore
    fetch_treasury_par_curve = None  # type: ignore


# Canonical tenor configuration
TENOR_ORDER = ["1M","3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","20Y","30Y"]
TENOR_YEARS = {"1M":1/12,"3M":3/12,"6M":6/12,"1Y":1,"2Y":2,"3Y":3,"5Y":5,"7Y":7,"10Y":10,"20Y":20,"30Y":30}
TENOR_TO_SERIES = {
    "1M": "DGS1MO","3M": "DGS3MO","6M": "DGS6MO","1Y": "DGS1","2Y": "DGS2","3Y": "DGS3","5Y": "DGS5","7Y": "DGS7","10Y": "DGS10","20Y": "DGS20","30Y": "DGS30",
}

# Map bondviz BC_* columns to canonical tenor labels used in this explorer
BC_TO_TENOR = {
    "BC_1MONTH": "1M",
    "BC_2MONTH": "2M",  # not plotted by default but can be kept
    "BC_3MONTH": "3M",
    "BC_6MONTH": "6M",
    "BC_1YEAR": "1Y",
    "BC_2YEAR": "2Y",
    "BC_3YEAR": "3Y",
    "BC_5YEAR": "5Y",
    "BC_7YEAR": "7Y",
    "BC_10YEAR": "10Y",
    "BC_20YEAR": "20Y",
    "BC_30YEAR": "30Y",
}


def _lazy_import_pdr():  # pragma: no cover - optional dependency
    global pdr  # type: ignore
    try:
        return pdr  # type: ignore
    except NameError:
        from pandas_datareader import data as pdr  # type: ignore
        return pdr


@st.cache_data(show_spinner=False)
def load_from_bondviz_range(start: Date, end: Date) -> pd.DataFrame:
    """Assemble a wide tenor DataFrame over a date range using bondviz fetchers.

    Accepts frames shaped like Treasury XML output (DATE + BC_* columns) and
    renames BC_* → canonical tenor labels (e.g., BC_1MONTH → 1M).
    Values are expected as percent levels.
    """
    if not USE_BONDVIZ or fetch_treasury_par_curve is None:
        return pd.DataFrame()

    years = list(range(start.year, end.year + 1))
    frames: list[pd.DataFrame] = []
    for y in years:
        try:
            dfy = fetch_treasury_par_curve(y)
            if dfy is None or len(dfy) == 0:
                continue
            # Normalize date column case-insensitively
            cols_lower = {c.lower(): c for c in dfy.columns}
            date_col = cols_lower.get("date") or cols_lower.get("DATE".lower())
            if date_col is None:
                if not isinstance(dfy.index, pd.DatetimeIndex):
                    raise ValueError("bondviz frame missing Date/DATE column and not datetime-indexed")
            else:
                dfy[date_col] = pd.to_datetime(dfy[date_col])
                dfy = dfy.set_index(date_col)

            # Rename BC_* columns to canonical tenor labels if present
            rename_bc = {c: BC_TO_TENOR[c] for c in dfy.columns if c in BC_TO_TENOR}
            if rename_bc:
                dfy = dfy.rename(columns=rename_bc)

            frames.append(dfy)
        except Exception as e:
            st.warning(f"bondviz year {y} failed: {e}")
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=0).sort_index()
    cols = [c for c in TENOR_ORDER if c in df.columns]
    if not cols:
        # Try to coerce columns like '1 Mo', '3 Mo' etc.
        rename_map: dict[str, str] = {}
        for c in df.columns:
            c_clean = str(c).upper().replace(" ", "")
            if c_clean.endswith("MO") and (c_clean[:-2] + "M") in TENOR_ORDER:
                rename_map[c] = c_clean[:-2] + "M"
            elif c_clean.endswith("Y") and c_clean in TENOR_ORDER:
                rename_map[c] = c_clean
        if rename_map:
            df = df.rename(columns=rename_map)
            cols = [c for c in TENOR_ORDER if c in df.columns]
    if not cols:
        return pd.DataFrame()

    out = df[cols].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")
    out = out.loc[(out.index >= pd.to_datetime(start)) & (out.index <= pd.to_datetime(end))]
    return out


@st.cache_data(show_spinner=False)
def load_from_fred(series_map: dict, start: Date, end: Date) -> pd.DataFrame:  # pragma: no cover
    pdr_local = _lazy_import_pdr()
    frames = []
    for tenor, fred_code in series_map.items():
        try:
            s = pdr_local.DataReader(fred_code, "fred", start, end)
            s.columns = [tenor]
            frames.append(s)
        except Exception as e:
            st.warning(f"FRED load failed for {tenor} ({fred_code}): {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).sort_index()
    return df.apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")


@st.cache_data(show_spinner=False)
def load_from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    norm_cols = {str(c).strip(): c for c in df.columns}
    if "Date" in norm_cols:
        base = norm_cols["Date"]
    elif "DATE" in norm_cols:
        base = norm_cols["DATE"]
    else:
        raise ValueError("CSV must include a 'Date' or 'DATE' column")
    df[base] = pd.to_datetime(df[base])
    df = df.set_index(base).sort_index()
    cols = [c for c in TENOR_ORDER if c in df.columns]
    if not cols:
        raise ValueError("CSV missing tenor columns. Expected some of: " + ", ".join(TENOR_ORDER))
    return df[cols].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")


def _latest_non_na_date(frame: pd.DataFrame) -> pd.Timestamp:
    mask = frame.notna().sum(axis=1) >= max(3, frame.shape[1] // 2)
    return frame[mask].index.max()


def _nearest_date(target: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    if target < index.min():
        return index.min()
    if target > index.max():
        return index.max()
    pos = index.get_indexer([target], method="nearest")[0]
    return index[pos]


def render_visualizer():
    """Render the Treasury Yield Curve Visual Explorer inside the current Streamlit app."""
    st.title("US Treasury Yield Curve – Visual Explorer (bondviz-integrated)")
    st.markdown(
        """
This app is wired to your bondviz fetchers first, and falls back to FRED only if needed. It provides:
- Current yield curve (by tenor)
- Curve shifts vs 1M/3M/6M/1Y ago
- Spreads: 2s10s and 3m10y
- Heatmap of yields over time
- Optional zero curve and 1Y forward (illustrative)

Fix included: gracefully handles Date vs DATE to avoid KeyError.
        """
    )

    # Sidebar controls (in a form to prevent constant reruns)
    with st.sidebar:
        source_options = ["bondviz API", "FRED API", "Upload CSV"] if USE_BONDVIZ else ["FRED API", "Upload CSV"]
        default_start = date.today() - timedelta(days=365 * 5)
        with st.form("viz_controls"):
            st.header("Data Source")
            st.radio("Choose input", options=source_options, index=0, key="viz_source")

            st.header("Date Range")
            st.date_input("Start", default_start, key="viz_start")
            st.date_input("End", date.today(), key="viz_end")

            st.header("Smoothing")
            st.checkbox("Apply 5-day rolling average", value=True, key="viz_smooth")
            submitted_controls = st.form_submit_button("Update Charts")

        # Optional upload field (outside form is OK; upload itself triggers a rerun once)
        uploaded = st.file_uploader("Upload CSV (Date + tenor columns)", type=["csv"], key="viz_csv")

    # Resolve control values from session_state (stable until submit)
    source = st.session_state.get("viz_source", source_options[0])
    start = st.session_state.get("viz_start", default_start)
    end = st.session_state.get("viz_end", date.today())
    smooth = st.session_state.get("viz_smooth", True)

    # Load data per user choice
    if source == "bondviz API":
        df = load_from_bondviz_range(start, end)
        if df.empty:
            st.warning("No data from bondviz source in the selected range.")
            st.stop()
    elif source == "FRED API":
        df = load_from_fred(TENOR_TO_SERIES, start, end)
        if df.empty:
            st.stop()
    else:
        if not uploaded:
            st.info("Upload a CSV to continue.")
            st.stop()
        df = load_from_csv(uploaded)

    if smooth and len(df) > 5:
        df = df.rolling(5, min_periods=1).mean()

    st.subheader("Data Snapshot")
    st.dataframe(df.tail(10))

    # Prepare latest and comparison curves
    latest_date = _latest_non_na_date(df)
    latest_curve = df.loc[latest_date].dropna()
    comparisons = {
        "1M ago": latest_date - pd.DateOffset(months=1),
        "3M ago": latest_date - pd.DateOffset(months=3),
        "6M ago": latest_date - pd.DateOffset(months=6),
        "1Y ago": latest_date - pd.DateOffset(years=1),
    }
    comp_curves = {}
    for label, t in comparisons.items():
        d = _nearest_date(pd.Timestamp(t), df.index)
        comp_curves[label] = df.loc[d].dropna()

    # Plot 1: Latest Yield Curve
    st.subheader("Latest Yield Curve")
    fig1, ax1 = plt.subplots()
    mat = [TENOR_YEARS[t] for t in latest_curve.index]
    ax1.plot(mat, latest_curve.values, marker="o")
    ax1.set_xlabel("Maturity (years)")
    ax1.set_ylabel("Yield (%)")
    ax1.set_title(f"Yield Curve on {latest_date.date()}")
    ax1.grid(True, which="both", linestyle=":")
    st.pyplot(fig1)

    # Plot 2: Curve Shifts vs Past
    st.subheader("Curve Shifts vs Past Dates")
    fig2, ax2 = plt.subplots()
    ax2.plot([TENOR_YEARS[t] for t in latest_curve.index], latest_curve.values, marker="o", label=str(latest_date.date()))
    for label, series in comp_curves.items():
        ax2.plot([TENOR_YEARS[t] for t in series.index], series.values, marker="o", label=label)
    ax2.set_xlabel("Maturity (years)")
    ax2.set_ylabel("Yield (%)")
    ax2.set_title("Yield Curve – Latest vs Past")
    ax2.grid(True, which="both", linestyle=":")
    ax2.legend()
    st.pyplot(fig2)

    # Plot 3: Spreads
    st.subheader("Key Spreads")
    needed = ["2Y", "10Y", "3M"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.info("Spreads require 3M, 2Y, and 10Y to be present.")
    else:
        spreads = pd.DataFrame(index=df.index)
        spreads["2s10s"] = df["10Y"] - df["2Y"]
        spreads["3m10y"] = df["10Y"] - df["3M"]

        fig3, ax3 = plt.subplots()
        ax3.plot(spreads.index, spreads["2s10s"] * 100, label="2s10s")
        ax3.plot(spreads.index, spreads["3m10y"] * 100, label="3m10y")
        ax3.axhline(0, linestyle=":")
        ax3.set_title("Spreads Over Time")
        ax3.set_ylabel("Spread (bps)")
        ax3.grid(True, which="both", linestyle=":")
        ax3.legend()
        st.pyplot(fig3)

    # Plot 4: Heatmap
    st.subheader("Yield Heatmap (Time × Tenor)")
    mat_cols = [c for c in TENOR_ORDER if c in df.columns]
    Z = df[mat_cols].to_numpy()
    Y = np.arange(Z.shape[0])
    X = np.arange(Z.shape[1])
    fig4, ax4 = plt.subplots()
    c = ax4.pcolormesh(X, Y, Z, shading="auto")
    ax4.set_title("Yields (%)")
    ax4.set_xlabel("Tenor")
    ax4.set_ylabel("Time (old → new)")
    ax4.set_xticks(np.arange(len(mat_cols)) + 0.5)
    ax4.set_xticklabels(mat_cols, rotation=0)
    fig4.colorbar(c, ax=ax4, label="%")
    st.pyplot(fig4)

    # Bootstrapped zero and forward curves from par yields
    st.subheader("Bootstrapped Zero Curve & 1Y Forward (semiannual)")
    st.caption("Zeros bootstrapped from par yields (semiannual coupons); forwards from discount factors.")

    def bootstrap_zeros_from_par(par_yields_pct: pd.Series) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Bootstrap discount factors and zero rates from par yields using semiannual coupons.
        Returns (grid_times_years, zero_rates_annual, df_map) where df_map maps T->DF.
        """
        y = (par_yields_pct / 100.0).dropna()
        if y.empty:
            return np.array([]), np.array([]), {}
        # Known par nodes (T in years, r in decimals)
        known_T = np.array([TENOR_YEARS[t] for t in y.index])
        known_r = y.values
        order = np.argsort(known_T)
        known_T = known_T[order]
        known_r = known_r[order]

        max_T = known_T.max()
        # Build semiannual grid from 0.5Y up to max_T
        f = 2  # semiannual
        grid = np.arange(0.5, max_T + 1e-9, 0.5)
        # Interpolate par yields onto the grid (flat extrapolation at ends)
        r_grid = np.interp(grid, known_T, known_r)

        D: dict[float, float] = {}
        for T, r_par in zip(grid, r_grid):
            n = int(round(T * f))
            c = r_par / f
            if n == 1:
                # One coupon/payment: price 1 = (1+c)*D(T)
                D[T] = 1.0 / (1.0 + c)
            else:
                # 1 = c*sum_{k=1..n-1} D(k/f) + (1+c) D(T)
                s = 0.0
                for k in range(1, n):
                    Tk = k / f
                    # Ensure previous DF exists (should by construction); if not, linearly interpolate
                    if Tk in D:
                        s += D[Tk]
                    else:
                        # Linear interpolation between nearest available nodes in D
                        prev_keys = sorted([tt for tt in D.keys() if tt < Tk])
                        next_keys = sorted([tt for tt in D.keys() if tt > Tk])
                        if prev_keys and next_keys:
                            t0 = prev_keys[-1]; t1 = next_keys[0]
                            Dk = D[t0] + (D[t1] - D[t0]) * (Tk - t0) / (t1 - t0)
                        elif prev_keys:
                            Dk = D[prev_keys[-1]]
                        elif next_keys:
                            Dk = D[next_keys[0]]
                        else:
                            Dk = np.exp(-r_par * Tk)  # fallback
                        s += Dk
                D[T] = max(1e-9, (1.0 - c * s) / (1.0 + c))

        # Convert DFs to annual-compounded zero rates: (1/D)^(1/T) - 1
        zeros = np.array([(D[T] ** (-1.0 / T)) - 1.0 for T in grid])
        return grid, zeros, D

    if not latest_curve.empty:
        grid, zero_rates, Dmap = bootstrap_zeros_from_par(latest_curve)
        if grid.size:
            fig5, ax5 = plt.subplots()
            ax5.plot(grid, zero_rates * 100)
            ax5.set_title(f"Bootstrapped Zero Curve – {latest_date.date()}")
            ax5.set_xlabel("Maturity (years)")
            ax5.set_ylabel("Zero Rate (annual, %)")
            ax5.grid(True, linestyle=":")
            st.pyplot(fig5)

            # 1Y forward starting at t: f(t,t+1) = D(t)/D(t+1) - 1 (annualized)
            t_vals = []
            fwd_vals = []
            for T in grid:
                T2 = T + 1.0
                if any(abs(T2 - g) < 1e-9 for g in grid):
                    # Use exact grid if present
                    D_t = Dmap[T]
                    D_t1 = Dmap[T2]
                    fwd = (D_t / D_t1) - 1.0
                    t_vals.append(T)
                    fwd_vals.append(fwd)
            if t_vals:
                fig6, ax6 = plt.subplots()
                ax6.plot(t_vals, np.array(fwd_vals) * 100)
                ax6.set_title("Bootstrapped 1Y Forward Rate Curve (from t to t+1)")
                ax6.set_xlabel("Start (years)")
                ax6.set_ylabel("Forward 1Y (annual, %)")
                ax6.grid(True, linestyle=":")
                st.pyplot(fig6)

    st.success("Loaded via: " + ("bondviz API" if source == "bondviz API" else source))
