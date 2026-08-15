"""
Streamlit interface for the shipping-cost prediction pipeline. Lets you configure a run,
execute it, and browse results without touching the CLI.

Run: streamlit run streamlit_app.py
"""
import time

import pandas as pd
import streamlit as st

from config import get_config
from data_generator import generate
from evaluate import run_comparison
from feature_importance import shap_importance
from feature_selection import build_feature_branches
from preprocessing import build_variants

st.set_page_config(page_title="Shipping Cost Pipeline", page_icon="\U0001F69A", layout="wide")

st.title("Freight Shipping-Cost Prediction — Pipeline Dashboard")
st.caption(
    "Reproduction of Jang, Chang & Kim (2023), *Prediction of Shipping Cost on Freight "
    "Brokerage Platform Using Machine Learning*, Sustainability 15(2):1122, on **synthetic "
    "data** matching the paper's schema (the real dataset is private/proprietary)."
)

with st.sidebar:
    st.header("Run configuration")
    n_rows = st.slider("Synthetic rows", 2000, 50000, 8000, step=1000)
    faithful = st.toggle(
        "Faithful mode (paper's literal protocol)",
        value=False,
        help="30-fold CV, 500-epoch DNN, literal boosting hyperparameters. Much slower.",
    )
    st.divider()
    use_real = st.toggle(
        "Use real data instead of synthetic",
        value=False,
        help="Runs on the SCMS supply-chain shipment dataset (real, public, ~10K rows) "
             "instead of the synthetic generator. See real_data.py.",
    )
    real_data_path = "data/SCMS_Delivery_History_Dataset.csv"
    if use_real:
        st.caption(f"Loading: `{real_data_path}`")
    run_button = st.button("Run pipeline", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = None

if run_button:
    config = get_config(faithful=faithful)
    config = config.__class__(
        data=config.data.__class__(n_rows=n_rows, seed=config.data.seed),
        preprocess=config.preprocess,
        feature_selection=config.feature_selection,
        validation=config.validation,
    )

    progress = st.progress(0, text="Loading data...")
    if use_real:
        from real_data import load_real_data, real_continuous_columns
        raw = load_real_data(real_data_path)
        continuous_cols = real_continuous_columns()
    else:
        raw = generate(config.data)
        continuous_cols = None
    progress.progress(20, text="Preprocessing...")
    variants = build_variants(raw, config.preprocess, continuous_columns=continuous_cols)
    progress.progress(40, text="Selecting features...")
    branches = build_feature_branches(variants, config.feature_selection)
    progress.progress(60, text="Training & evaluating models (this is the slow part)...")
    start = time.time()
    results = run_comparison(variants, branches, config.validation)
    elapsed = time.time() - start
    progress.progress(90, text="Computing SHAP feature importance...")
    step_branch = branches.get(("stepwise", "listwise_deletion")) or next(iter(branches.values()))
    importance, share = shap_importance(variants["listwise_deletion"], step_branch["features"])
    progress.progress(100, text="Done.")

    st.session_state.results = results
    st.session_state.importance = importance
    st.session_state.share = share
    st.session_state.elapsed = elapsed
    st.session_state.n_rows = n_rows
    st.session_state.faithful = faithful
    st.session_state.use_real = use_real

if st.session_state.results is not None:
    results: pd.DataFrame = st.session_state.results
    share = st.session_state.share

    best = results.loc[results["r_squared"].idxmax()]
    boosting_best = results[results["model"].isin(["XGBoost", "LightGBM"])]["r_squared"].max()
    mlr_best = results[results["model"] == "MLR"]["r_squared"].max()

    st.success(
        f"Ran on {'real SCMS data' if st.session_state.use_real else f'{st.session_state.n_rows:,} synthetic rows'} "
        f"in {st.session_state.elapsed:.1f}s ({'faithful' if st.session_state.faithful else 'demo'} protocol)."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best model", best["model"])
    col2.metric("Best R²", f"{best['r_squared']:.3f}")
    col3.metric("Top feature", share.index[0], f"{share.iloc[0]:.1f}% SHAP share")
    col4.metric(
        "Boosting vs. MLR",
        f"{boosting_best:.3f} vs {mlr_best:.3f}",
        "boosting ahead" if boosting_best > mlr_best else "MLR ahead",
    )

    st.subheader("Model comparison (cf. paper Table 14)")
    pivot = results.pivot(index="model", columns="branch", values="r_squared")
    st.bar_chart(pivot.T)
    st.dataframe(pivot.style.format("{:.3f}").background_gradient(cmap="YlOrBr", axis=None),
                 use_container_width=True)

    st.subheader("Feature importance (SHAP, cf. paper Figure 3)")
    st.bar_chart(share.sort_values())

    st.subheader("Raw results")
    st.dataframe(results, use_container_width=True)
    st.download_button(
        "Download comparison_table.csv", results.to_csv(index=False),
        file_name="comparison_table.csv", mime="text/csv",
    )
else:
    st.info("Configure a run in the sidebar and click **Run pipeline** to see results here.")
