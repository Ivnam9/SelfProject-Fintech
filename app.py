"""
Streamlit Dashboard — Yield Curve & Fixed Income Analytics Engine
-------------------------------------------------------------------
Run with: streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from bond_pricer import Bond
from data_loader import load_treasury_yields, MATURITIES_YEARS, MATURITY_LABELS
from nss_model import fit_nss, fit_nss_panel
from pca_analysis import run_pca_on_yield_changes, interpret_components
from regime_classifier import build_features, fit_hmm_regimes, label_states, threshold_classify

st.set_page_config(page_title="Fixed Income Analytics Engine", layout="wide")
st.title("Yield Curve & Fixed Income Analytics Engine")

DATA_SOURCE_NOTE = (
    "⚠️ Data source: SYNTHETIC panel (this sandbox has no live FRED/RBI network access). "
    "Set `FRED_API_KEY` and toggle live mode in `data_loader.py` to use real US Treasury data."
)
st.info(DATA_SOURCE_NOTE)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Bond Pricer", "Yield Curve (NSS)", "PCA Decomposition", "Regime Detector"]
)

# ---------------- Tab 1: Bond Pricer ----------------
with tab1:
    st.header("Module 1 — Bond Pricing Engine")
    col1, col2 = st.columns(2)
    with col1:
        face = st.number_input("Face value", value=100.0)
        coupon = st.number_input("Annual coupon rate (%)", value=6.0) / 100
        maturity = st.number_input("Years to maturity", value=10.0)
        freq = st.selectbox("Coupon frequency", [2, 1], format_func=lambda x: "Semi-annual" if x == 2 else "Annual")
    with col2:
        ytm_input = st.number_input("Yield to maturity (%) — for pricing", value=8.0) / 100

    bond = Bond(face, coupon, maturity, freq)
    price = bond.clean_price(ytm_input)
    mac_dur = bond.macaulay_duration(ytm_input)
    mod_dur = bond.modified_duration(ytm_input)
    conv = bond.convexity(ytm_input)
    dv01 = bond.dv01(ytm_input)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Clean Price", f"{price:.4f}")
    m2.metric("Macaulay Duration", f"{mac_dur:.3f} yrs")
    m3.metric("Modified Duration", f"{mod_dur:.3f}")
    m4.metric("Convexity", f"{conv:.3f}")
    m5.metric("DV01", f"{dv01:.5f}")

    st.caption("Validated against a known textbook example (10Y, 6% semi-annual coupon @ 8% YTM → 86.41). "
               "See tests/test_bond_pricer.py for the full validation suite (7/7 passing).")

# ---------------- Load shared data for tabs 2-4 ----------------
@st.cache_data
def get_data():
    return load_treasury_yields(use_live=True, start="2015-01-01", end="2024-12-31")

df = get_data()

# ---------------- Tab 2: NSS Curve ----------------
with tab2:
    st.header("Module 2 — Nelson-Siegel-Svensson Curve Fitting")
    date_choice = st.select_slider("Select date", options=list(df.index), value=df.index[-1])
    row = df.loc[date_choice]

    res = fit_nss(MATURITIES_YEARS, row.values)
    fig, ax = plt.subplots()
    ax.scatter(MATURITIES_YEARS, row.values, label="Observed", color="black", zorder=5)
    t_grid = np.linspace(0.05, 30, 200)
    from nss_model import nss_yield
    p = res["params"]
    fitted_curve = nss_yield(t_grid, p["beta0"], p["beta1"], p["beta2"], 0.0, p["tau1"], p["tau2"])
    ax.plot(t_grid, fitted_curve, label="NSS fit", color="tab:blue")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (%)")
    ax.legend()
    ax.set_title(f"NSS fit — {date_choice.date()} (RMSE = {res['rmse_bp']:.2f} bp)")
    st.pyplot(fig)

    st.caption("Target from project plan: RMSE < 5 bp. Average across the full sample is reported below.")
    if st.button("Run full-panel NSS fit (756 days)"):
        with st.spinner("Fitting NSS to every day..."):
            params_df, rmse_series = fit_nss_panel(df, MATURITIES_YEARS)
        st.success(f"Mean RMSE: {rmse_series.mean():.3f} bp | Median: {rmse_series.median():.3f} bp | Max: {rmse_series.max():.3f} bp")
        st.line_chart(rmse_series)

# ---------------- Tab 3: PCA ----------------
with tab3:
    st.header("Module 3 — PCA on Yield Curve Dynamics")
    pca_result = run_pca_on_yield_changes(df)
    var_explained = pca_result["explained_variance_ratio"]
    cum_var = pca_result["cumulative_variance"]

    c1, c2, c3 = st.columns(3)
    c1.metric("PC1 (level)", f"{var_explained[0]*100:.1f}%")
    c2.metric("PC2 (slope)", f"{var_explained[1]*100:.1f}%")
    c3.metric("PC3 (curvature)", f"{var_explained[2]*100:.1f}%")
    st.metric("Cumulative variance (PC1-3)", f"{cum_var[-1]*100:.1f}%")

    fig2, ax2 = plt.subplots()
    pca_result["loadings"].plot(ax=ax2, marker="o")
    ax2.set_xticks(range(len(MATURITY_LABELS)))
    ax2.set_xticklabels(MATURITY_LABELS)
    ax2.set_title("PCA Factor Loadings by Maturity")
    ax2.axhline(0, color="gray", lw=0.5)
    st.pyplot(fig2)

    st.write("Interpretation:", interpret_components(pca_result["loadings"]))

# ---------------- Tab 4: Regime Detector ----------------
with tab4:
    st.header("Module 4 — Regime Detection")
    feats = build_features(df)
    model, states = fit_hmm_regimes(feats, n_states=4)
    labels = label_states(feats, states, n_states=4)
    feats = feats.copy()
    feats["hmm_regime"] = pd.Series(states, index=feats.index).map(labels)
    feats["rule_regime"] = threshold_classify(feats)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(df.index[-len(feats):], feats["level"], color="black", lw=1)
    regime_colors = {"Rising Rates": "red", "Falling Rates": "green",
                      "Inverted": "purple", "Flat / Range-bound": "gray"}
    for regime, color in regime_colors.items():
        mask = feats["hmm_regime"] == regime
        ax3.scatter(feats.index[mask], feats["level"][mask], color=color, s=8, label=regime)
    ax3.legend()
    ax3.set_title("10Y Yield with HMM-Detected Regimes")
    st.pyplot(fig3)

    agreement = (feats["hmm_regime"] == feats["rule_regime"]).mean()
    st.metric("HMM vs. rule-based agreement", f"{agreement*100:.1f}%")
    st.bar_chart(feats["hmm_regime"].value_counts())
