"""
Nelson-Siegel-Svensson (NSS) Curve Fitting
--------------------------------------------
Fits the 6-parameter NSS model to a daily observed yield curve via
scipy.optimize.minimize (L-BFGS-B), and reports fit quality (RMSE in bp).

NSS formula:
y(t) = b0 + b1*(1-e^(-t/tau1))/(t/tau1)
          + b2*((1-e^(-t/tau1))/(t/tau1) - e^(-t/tau1))
          + b3*((1-e^(-t/tau2))/(t/tau2) - e^(-t/tau2))
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def nss_yield(t, beta0, beta1, beta2, beta3, tau1, tau2):
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-t / tau1)) / (t / tau1)
    term3 = beta2 * ((1 - np.exp(-t / tau1)) / (t / tau1) - np.exp(-t / tau1))
    term4 = beta3 * ((1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2))
    return term1 + term2 + term3 + term4


def fit_nss(maturities, observed_yields, x0=None):
    """
    maturities       : array of maturities in years
    observed_yields  : array of observed par yields (%), same length
    Returns dict with fitted params, fitted curve, and RMSE in basis points.
    """
    maturities = np.asarray(maturities, dtype=float)
    observed = np.asarray(observed_yields, dtype=float)

    if x0 is None:
        x0 = np.array([observed[-1], observed[0] - observed[-1], 0.0, 1.5, 6.0])

    def objective(params):
        b0, b1, b2, tau1, tau2 = params
        fitted = nss_yield(maturities, b0, b1, b2, 0.0, tau1, tau2)
        return np.sum((fitted - observed) ** 2)

    bounds = [(-5, 15), (-15, 15), (-30, 30), (0.05, 10), (0.05, 30)]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    b0, b1, b2, tau1, tau2 = result.x
    fitted = nss_yield(maturities, b0, b1, b2, 0.0, tau1, tau2)
    rmse_bp = np.sqrt(np.mean((fitted - observed) ** 2)) * 100  # % -> bp

    return {
        "params": {"beta0": b0, "beta1": b1, "beta2": b2, "tau1": tau1, "tau2": tau2},
        "fitted_yields": fitted,
        "rmse_bp": rmse_bp,
        "success": result.success,
    }


def fit_nss_panel(yield_df, maturities_years):
    """
    Fits NSS to every row (day) of a yield panel DataFrame.
    Returns (params_df, rmse_series).
    """
    records = []
    rmses = []
    x0 = None
    for date, row in yield_df.iterrows():
        res = fit_nss(maturities_years, row.values, x0=x0)
        records.append({"date": date, **res["params"]})
        rmses.append(res["rmse_bp"])
        # warm-start next day with today's fitted params for stability/speed
        p = res["params"]
        x0 = np.array([p["beta0"], p["beta1"], p["beta2"], p["tau1"], p["tau2"]])

    params_df = pd.DataFrame(records).set_index("date")
    rmse_series = pd.Series(rmses, index=yield_df.index, name="rmse_bp")
    return params_df, rmse_series


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_treasury_yields, MATURITIES_YEARS

    df = load_treasury_yields(use_live=True, start="2015-01-01",   end="2024-12-31")
    params_df, rmse_series = fit_nss_panel(df, MATURITIES_YEARS)

    print(f"Fitted NSS on {len(df)} days.")
    print(f"Mean fit RMSE: {rmse_series.mean():.3f} bp")
    print(f"Median fit RMSE: {rmse_series.median():.3f} bp")
    print(f"Max fit RMSE: {rmse_series.max():.3f} bp")
    print(params_df.head())
