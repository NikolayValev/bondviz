from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from bondviz.visualizer import load_from_bondviz_range, load_from_fred, USE_BONDVIZ, TENOR_TO_SERIES

MACRO_INFLATION_SERIES = {"5Y5Y Inflation (FRED)": "T5YIFR"}
PCA_CACHE_KEY = "yield_pca_results"


@st.cache_resource(show_spinner=False)
def _require_sklearn():
    try:
        from sklearn.decomposition import PCA  # type: ignore
        return PCA
    except Exception:
        # Do not raise to avoid breaking the whole page; caller can handle None
        return None


def _read_yield_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Normalize column names and date handling
    cols_upper = {c: str(c).strip().upper() for c in df.columns}
    df = df.rename(columns=cols_upper)

    # Identify date column
    date_col = None
    for cand in ("DATE", "Date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # Try index if looks like date
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:  # keep original
            pass
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)

    df = df.sort_index()
    return df


@st.cache_data(show_spinner=False)
def _load_inflation_expectations(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    df = load_from_fred(MACRO_INFLATION_SERIES, start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float)
    series = df.iloc[:, 0].astype(float)
    series.name = df.columns[0]
    return series


def _parse_probability_csv(file) -> pd.Series:
    try:
        df = pd.read_csv(file)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    prob_col = next((c for c in df.columns if "prob" in c.lower()), None)
    if date_col is None or prob_col is None:
        return pd.Series(dtype=float)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    series = pd.to_numeric(df[prob_col], errors="coerce")
    series = pd.Series(series.values, index=df[date_col], name=df.columns[df.columns.get_loc(prob_col)])
    series = series.dropna()
    return series


def _zscore(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return series.copy()
    std = float(clean.std(ddof=0))
    if std == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index, name=series.name)
    mean = float(clean.mean())
    return (series - mean) / std


def render_yield_pca():
    st.subheader("Yield Curve PCA")
    with st.form("pca_form"):
        mode = st.radio("Source", ["Treasury Data", "Upload CSV"], horizontal=True)
        n_components = st.number_input("Principal components", value=3, min_value=1, max_value=6, step=1)

        df: Optional[pd.DataFrame] = None
        if mode == "Treasury Data":
            use_current = st.checkbox(
                "Use data loaded in Treasury Yields page (if available)", value=True, key="pca_use_current"
            )
            if not use_current or "yield_curve_df" not in st.session_state:
                # Minimal fetch controls
                c1, c2 = st.columns(2)
                with c1:
                    default_start = pd.to_datetime("today") - pd.DateOffset(years=2)
                    start = st.date_input("Start", default_start.date())
                with c2:
                    end = st.date_input("End", pd.to_datetime("today").date())
                source_options = ["bondviz API", "FRED API"] if USE_BONDVIZ else ["FRED API"]
                source = st.selectbox("Source", source_options, index=0)
            else:
                start = end = None  # type: ignore
                source = None  # type: ignore
        else:
            uploaded = st.file_uploader("Upload CSV (must include a Date/DATE column and tenor columns)", type=["csv"], key="pca_csv")
            start = end = None  # type: ignore
            source = None  # type: ignore

        submitted = st.form_submit_button("Run PCA")

    cache_payload = st.session_state.get(PCA_CACHE_KEY)

    # Lightweight formatters for narratives
    def _fmt_pct(value: float | None, decimals: int = 1) -> str:
        if value is None or np.isnan(value):
            return "n/a"
        return f"{value:.{decimals}f}%"

    def _describe_loading(vec: np.ndarray) -> str:
        if vec.size == 0 or np.isnan(vec).all():
            return "mixed influence across the curve"
        front = float(np.nanmean(vec[: max(1, vec.size // 3)]))
        back = float(np.nanmean(vec[-max(1, vec.size // 3):]))
        sign = np.sign(np.nanmean(vec))
        if abs(front) < 1e-6 and abs(back) < 1e-6:
            return "negligible loading across most maturities"
        if sign >= 0:
            if front > back:
                return "heavier weight on the front end"
            if back > front:
                return "heavier weight on the long end"
            return "mostly parallel impact across maturities"
        else:
            if front < back:
                return "steepening move (front vs long signs differ)"
            if back < front:
                return "flattening move (long end opposite front)"
            return "counter-parallel influence across maturities"

    def _describe_factor_path(series: pd.Series) -> str:
        if series.empty or series.dropna().empty:
            return "insufficient history to gauge the trend"
        recent = series.dropna().tail(min(60, len(series)))
        change = float(recent.iloc[-1] - recent.iloc[0])
        last = float(recent.iloc[-1])
        if abs(change) < 0.25:
            direction = "has been broadly stable"
        elif change > 0:
            direction = "has been trending higher"
        else:
            direction = "has been rolling over"
        bias = "slightly positive" if last > 0.5 else "slightly negative" if last < -0.5 else "near neutral"
        return f"{direction}; latest reading sits {bias}."

    @st.cache_data(show_spinner=False)
    def _run_pca_cached(X: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, list]:
        pca = PCA(n_components=int(n))
        factors = pca.fit_transform(X.values)
        factor_cols = [f"PC{i+1}" for i in range(int(n))]
        factor_df = pd.DataFrame(factors, index=X.index, columns=factor_cols)
        explained = (pca.explained_variance_ratio_ * 100.0)
        components = pca.components_.copy()
        cols = list(X.columns)
        return factor_df, explained, components, cols

    if submitted:
        PCA = _require_sklearn()
        if PCA is None:
            st.warning("scikit-learn is not installed. Install with `pip install scikit-learn` and rerun.")
            return

        if mode == "Treasury Data":
            if "yield_curve_df" in st.session_state and st.session_state.get("pca_use_current", True):
                # Use the dataset saved by the visualizer if available
                df = st.session_state.get("yield_curve_df")
            else:
                if source == "bondviz API":
                    df = load_from_bondviz_range(start, end)  # type: ignore
                else:
                    df = load_from_fred(TENOR_TO_SERIES, start, end)  # type: ignore
        else:
            if uploaded is None:
                st.error("Please upload a CSV file.")
                return
            df = _read_yield_csv(uploaded)

        if df is None or df.empty:
            st.error("No data available for PCA.")
            return
        # Keep only numeric tenor columns
        yield_df = df.select_dtypes(include=[np.number]).dropna(how="all", axis=0)
        if yield_df.empty:
            st.error("No numeric tenor columns found after parsing the file.")
            return

        # Standardize across time (per tenor)
        X = (yield_df - yield_df.mean()) / yield_df.std(ddof=0)
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if X.empty:
            st.error("Insufficient data after standardization.")
            return

        with st.status("Running PCA...", expanded=False) as status:
            # Execute cached PCA (fast on subsequent runs); show a status animation
            factor_df, explained, components, cols = _run_pca_cached(X, int(n_components))
            status.update(label="PCA complete", state="complete", expanded=False)

        cache_payload = {
            "factor_df": factor_df,
            "explained": explained,
            "components": components,
            "cols": cols,
            "n_components": int(n_components),
        }
        st.session_state[PCA_CACHE_KEY] = cache_payload
    elif cache_payload is None:
        return

    factor_df = cache_payload["factor_df"]
    explained = cache_payload["explained"]
    components = cache_payload["components"]
    cols = cache_payload["cols"]
    n_components = int(cache_payload["n_components"])
    if not submitted:
        st.caption("Using cached PCA results. Re-run the form to refresh the dataset.")

    component_aliases = ["Level", "Slope", "Curvature", "Butterfly", "Residual"]
    component_names: dict[str, str] = {}
    highlight_lines: list[str] = []
    for i in range(int(n_components)):
        key = f"PC{i+1}"
        alias = component_aliases[i] if i < len(component_aliases) else f"Factor {i+1}"
        component_names[key] = alias
        explained_i = float(explained[i]) if i < len(explained) else float("nan")
        descriptor = _describe_loading(components[i]) if i < components.shape[0] else "no loading info"
        highlight_lines.append(
            f"- {alias} ({key}) captures {_fmt_pct(explained_i)} of the variance and shows {descriptor}."
        )
    total_pct = float(np.sum(explained[: int(n_components)])) if len(explained) else float("nan")
    highlight_lines.append(
        f"- Top {int(n_components)} components together explain {_fmt_pct(total_pct)} of the standardized yield-move variance."
    )

    st.markdown("### PCA Highlights")
    st.markdown("\n".join(highlight_lines))

    st.write("Explained variance (%):", np.round(explained, 2))

    # Plot 1: Loadings
    try:
        import matplotlib.pyplot as plt  # type: ignore
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        for i in range(int(n_components)):
            key = f"PC{i+1}"
            label = component_names.get(key, key)
            var_share = explained[i] if i < len(explained) else float("nan")
            ax1.plot(cols, components[i], marker="o", label=f"{label} ({var_share:.1f}%)")
        ax1.set_title("PCA Loadings - Yield Curve Factors")
        ax1.set_ylabel("Loading Weight")
        ax1.set_xlabel("Tenor")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend()
        st.pyplot(fig1)
        loading_notes = []
        for i in range(int(n_components)):
            key = f"PC{i+1}"
            alias = component_names.get(key, key)
            loading_notes.append(f"{alias}: {_describe_loading(components[i])}.")
        st.markdown("**Interpretation:** " + " ".join(loading_notes))
    except Exception as e:
        st.error(f"Failed to plot loadings: {e}")

    # Plot 2: Factor time series
    display_df = factor_df.rename(columns={
        f"PC{i+1}": f"{component_names.get(f'PC{i+1}', f'PC{i+1}')} (PC{i+1})"
        for i in range(int(n_components))
    })
    st.line_chart(display_df, height=320)
    factor_notes = []
    for i in range(min(3, int(n_components))):
        key = f"PC{i+1}"
        alias = component_names.get(key, key)
        factor_notes.append(f"{alias} {_describe_factor_path(factor_df[key])}")
    st.markdown("**Interpretation:** " + " ".join(factor_notes))

    st.markdown("### Macro Overlay Panel")
    overlay_options = [
        "None",
        "Inflation expectations (5y5y FRED)",
        "Fed-funds probabilities (upload CSV)",
    ]
    overlay_choice = st.selectbox("Contextual overlay", overlay_options, index=0, key="macro_overlay_choice")
    macro_series: Optional[pd.Series] = None
    macro_label = ""
    if overlay_choice != "None":
        history_start = factor_df.index.min()
        history_end = factor_df.index.max()
        if history_start is None or history_end is None:
            st.info("Run PCA first to build the macro overlay.")
        else:
            if overlay_choice.startswith("Inflation"):
                macro_series = _load_inflation_expectations(history_start, history_end)
                macro_label = macro_series.name or "Inflation"
                if macro_series.empty:
                    st.warning("Unable to load FRED inflation expectations. Ensure pandas-datareader is installed.")
            else:
                st.caption(
                    "Upload CME FedWatch (or similar) CSV with at least `Date` and `Probability` columns. "
                    "Probabilities can be in percent or decimal form."
                )
                fed_upload = st.file_uploader("Fed-funds probabilities", type=["csv"], key="fed_probs_csv")
                if fed_upload is not None:
                    macro_series = _parse_probability_csv(fed_upload)
                    macro_label = macro_series.name or "Fed-funds Probability"
                    if macro_series.empty:
                        st.warning("Could not parse the uploaded CSV; confirm it has Date and Probability columns.")
                else:
                    st.info("Provide a probability file export to unlock the overlay.")

    if macro_series is not None and not macro_series.empty:
        macro_series = macro_series.sort_index()
        selected_factors = st.multiselect(
            "PCA trend lines to compare",
            list(display_df.columns),
            default=list(display_df.columns[: min(2, len(display_df.columns))]),
            key="macro_overlay_factors",
        )
        if selected_factors:
            macro_norm = _zscore(macro_series)
            macro_norm.index = pd.to_datetime(macro_norm.index)
            # Align macro series to PCA calendar (forward-fill due to lower frequency macro data).
            macro_aligned = macro_norm.reindex(display_df.index, method="ffill").dropna()
            if macro_aligned.empty:
                st.warning("Macro series does not overlap with the PCA sample window.")
            else:
                panel = display_df[selected_factors].loc[macro_aligned.index].copy()
                macro_name = f"{macro_label} (z-score)"
                panel[macro_name] = macro_aligned
                st.line_chart(panel, height=360)
                correlations = []
                for col in selected_factors:
                    corr = panel[col].corr(panel[macro_name])
                    if corr == corr:  # skip NaN
                        correlations.append(f"{col} vs {macro_label}: {corr:+.2f}")
                explainer = (
                    "Overlay compares z-scores so PCA factors and macro series share a common scale."
                    " Macro values are forward-filled to match the PCA date grid."
                )
                st.caption(explainer)
                if correlations:
                    st.markdown("**Macro link:** " + "; ".join(correlations))
        else:
            st.info("Select at least one PCA factor to visualize alongside the macro series.")

    with st.expander("Show factor values"):
        st.dataframe(factor_df, use_container_width=True)
        st.caption("Factor realizations (z-scores) for each observation after de-meaning and scaling tenors.")
