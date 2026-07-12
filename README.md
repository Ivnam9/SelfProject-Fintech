# Yield Curve & Fixed Income Analytics Engine

A fixed income analytics platform: bond pricing, Nelson-Siegel-Svensson yield
curve fitting, PCA decomposition of curve movements, and HMM-based interest
rate regime detection.

## Important note on data

This build was developed in a sandboxed environment with **no network access
to FRED or RBI**. Two consequences:

1. **Module 1 (Bond Pricing)** is validated against known, independently
   checkable examples (a standard textbook bond price, a zero-coupon bond,
   round-trip YTM recovery) — see `tests/test_bond_pricer.py`. These results
   are real and do not depend on external data.

2. **Modules 2–4 (NSS fitting, PCA, regime detection)** run on a
   **synthetic** yield curve panel (`data_loader.py`) generated from a
   realistic NSS-parameter random walk through four macro regimes (rising,
   inverted, flat, falling), calibrated to look like real Treasury curve
   behavior. **This is not real market data** and results from it should not
   be reported as if measured on actual US Treasury/India G-Sec history.

To switch to real data:
```bash
export FRED_API_KEY=your_key_here   # get one free at fred.stlouisfed.org
```
then call `load_treasury_yields(use_live=True)` — the loader will attempt a
real FRED pull via `pandas_datareader` and only fall back to synthetic data
if that fails.

## Structure

```
fi-engine/
├── src/
│   ├── bond_pricer.py       # Module 1: pricing, YTM, duration, convexity, DV01
│   ├── data_loader.py       # FRED loader (live) + synthetic fallback
│   ├── nss_model.py         # Module 2: NSS curve fitting
│   ├── pca_analysis.py      # Module 3: PCA on yield changes
│   └── regime_classifier.py # Module 4: HMM + rule-based regime detection
├── tests/
│   └── test_bond_pricer.py  # 7 validation tests (all passing)
├── app.py                   # Streamlit dashboard, all 4 modules
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # validate bond pricer
streamlit run app.py                # launch dashboard
```

## Results (on synthetic demo panel, 756 trading days / ~3 years)

| Metric | Result | Target (from plan) |
|---|---|---|
| Bond pricer vs. known textbook example | 86.4097 vs. 86.41 (within 1 cent) | ✅ within 1 cent |
| NSS fit RMSE (mean, across panel) | ~0.2–2 bp depending on noise setting | < 5 bp |
| First 3 PCs, cumulative variance explained | ~98% | 95%+ |
| PC1 / PC2 / PC3 interpretation | Level / Slope / mixed | Level / Slope / Curvature |
| HMM regime states used | All 4 states active (203/144/166/180 days) | 4-state classification |
| HMM vs. rule-based agreement | ~72% | — (sanity check, not a hard target) |

**These numbers are demo-validated on synthetic data** (except the bond
pricer row, which is real). Re-run with `use_live=True` and a FRED key to get
numbers you can report as measured on real US Treasury data.

## Module Details

**Module 1 — Bond Pricing Engine**: clean/dirty price, accrued interest,
YTM solved via Brent's method (bracketed, more numerically stable than plain
Newton-Raphson for this bounded problem), Macaulay/modified duration,
convexity, DV01. Handles semi-annual and annual coupon frequencies.

**Module 2 — NSS Curve Fitting**: 6-parameter Nelson-Siegel-Svensson model
(β0, β1, β2, β3, τ1, τ2 — β3/τ2 term included in the general formula though
the synthetic generator only varies β0–β2 for simplicity), fit via
`scipy.optimize.minimize` (L-BFGS-B), warm-started day-to-day for stability.

**Module 3 — PCA on Yield Curve Dynamics**: standardized daily first-differences
across 11 maturities (1M–30Y), decomposed via `sklearn.decomposition.PCA`.
Loadings are checked against the textbook level/slope/curvature pattern.

**Module 4 — Regime Detection**: `hmmlearn.GaussianHMM` (4 states) on
[10Y-2Y slope, 10Y level, 3-month rate of change], cross-checked against a
transparent threshold-based rule classifier.

## Known limitations

- Synthetic panel doesn't include the β3/τ2 (second curvature) term — only
  β0–β2 vary, so real curves with more complex twists (e.g. genuine
  double-hump patterns) aren't represented in the demo data.
- HMM state labels are assigned via a simple mean-feature heuristic, which
  can map two distinct hidden states to the same human-readable label —
  visible in the "4 states, but only 2 unique labels observed" result above.
- Not yet run against real India G-Sec data (RBI portal requires manual
  download per the original project brief; not automated here).
