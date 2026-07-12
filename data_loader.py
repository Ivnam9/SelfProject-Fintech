"""
Data Loader
-----------
Primary path: pull real US Treasury par yield curve data from FRED
(requires a free API key from fred.stlouisfed.org).

Fallback path: this sandbox environment has no network access to FRED,
so a SYNTHETIC yield panel is generated instead, for demonstrating and
testing the NSS / PCA / regime modules end-to-end. The synthetic panel
is built from a realistic NSS parameterization + regime-switching drift
+ noise -- it is NOT real market data and must not be reported as such.

To use real data: set FRED_API_KEY as an environment variable and call
load_treasury_yields(use_live=True).
"""

import os
import numpy as np
import pandas as pd

MATURITIES_YEARS = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
MATURITY_LABELS = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

# FRED series IDs for on-the-run par yields (DGS = Daily Treasury par yield curve rates)
FRED_SERIES = {
    "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1",
    "2Y": "DGS2", "3Y": "DGS3", "5Y": "DGS5", "7Y": "DGS7",
    "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30",
}


def load_treasury_yields(use_live=False, start="2015-01-01", end="2024-12-31",
                          n_days_synthetic=756, seed=42):
    """
    Returns a DataFrame indexed by date, columns = maturity labels, values = yields (%).

    use_live=True attempts a real FRED pull (needs FRED_API_KEY env var + network access).
    Falls back to synthetic data with a clear warning if that fails or use_live=False.
    """
    if use_live:
        try:
            return _load_live_fred(start, end)
        except Exception as e:
            print(f"[data_loader] Live FRED fetch failed ({e}); falling back to synthetic data.")

    return _generate_synthetic_panel(n_days=n_days_synthetic, seed=seed)


def _load_live_fred(start, end):
    import pandas_datareader.data as web
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY environment variable not set")
    frames = {}
    for label, series_id in FRED_SERIES.items():
        s = web.DataReader(series_id, "fred", start, end, api_key=api_key)
        frames[label] = s[series_id]
    df = pd.DataFrame(frames).dropna(how="all")
    df.attrs["source"] = "FRED (live)"
    return df


def _nss_yield(t, beta0, beta1, beta2, beta3, tau1, tau2):
    t = np.maximum(t, 1e-6)
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-t / tau1)) / (t / tau1)
    term3 = beta2 * ((1 - np.exp(-t / tau1)) / (t / tau1) - np.exp(-t / tau1))
    term4 = beta3 * ((1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2))
    return term1 + term2 + term3 + term4


def _generate_synthetic_panel(n_days=756, seed=42):
    """
    Builds a SYNTHETIC daily yield curve panel (~3 trading years) by evolving
    NSS parameters through 4 macro regimes (rising / falling / flat / inverted),
    then adding small daily noise. This mimics realistic Treasury curve
    behavior (level/slope/curvature dynamics, occasional inversions) purely
    for module testing -- it is NOT sourced from FRED or any real dataset.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # Regime segments: (n_days_in_regime, beta0 target, beta1 target, beta2, tau1, tau2)
    # beta0 ~ long-run level, beta1 ~ (short - long) slope contribution, beta2 ~ curvature
    seg_len = max(1, n_days // 4)
    lengths = [seg_len, seg_len, seg_len, n_days - 3 * seg_len]
    seg_targets = [
        (1.5, 2.0, -1.0, 1.5, 8.0),    # rising rates (2022-style hiking)
        (4.5, -1.0, 1.0, 1.2, 6.0),    # inverted curve (short > long)
        (4.0, -0.3, 0.5, 1.5, 7.0),    # flat / peak
        (3.5, 1.5, -0.5, 2.0, 8.0),    # falling rates / re-steepening
    ]
    segments = [(lengths[i], *seg_targets[i]) for i in range(4)]

    rows = []
    beta_track = []
    day_idx = 0
    prev = np.array([1.5, 2.0, -1.0, 1.5, 8.0])
    for n_seg, b0, b1, b2, tau1, tau2 in segments:
        target = np.array([b0, b1, b2, tau1, tau2])
        for k in range(n_seg):
            frac = k / n_seg
            cur = prev + (target - prev) * frac
            beta0, beta1, tau1_, tau2_ = cur[0], cur[1], cur[3], cur[4]
            beta2_ = cur[2]
            # add small daily AR-ish noise to each beta (this is what drives real
            # day-to-day yield curve movement -- level/slope/curvature shifts,
            # not independent per-maturity noise)
            noise = rng.normal(0, 0.015, size=3)
            beta0 += noise[0]; beta1 += noise[1]; beta2_ += noise[2]
            yields = _nss_yield(np.array(MATURITIES_YEARS), beta0, beta1, beta2_, 0.0, tau1_, tau2_)
            # tiny idiosyncratic measurement noise per maturity (bid/ask, rounding) --
            # kept small relative to beta-driven moves so the panel stays low-rank,
            # matching how real Treasury curves behave
            yields += rng.normal(0, 0.003, size=len(yields))
            rows.append(yields)
            beta_track.append([beta0, beta1, beta2_, tau1_, tau2_])
        prev = target
        day_idx += n_seg

    df = pd.DataFrame(rows, index=dates[:len(rows)], columns=MATURITY_LABELS)
    df.attrs["source"] = "SYNTHETIC (no live FRED access in this environment)"
    return df


if __name__ == "__main__":
    df = load_treasury_yields(use_live=True)
    print(f"Loaded panel: {df.shape[0]} days x {df.shape[1]} maturities")
    print(f"Source: {df.attrs.get('source')}")
    print(df.head())
    print(df.tail())
