"""
PCA on Yield Curve Dynamics
----------------------------
Runs PCA on daily first-differences of the yield curve panel (T x N, N = maturities).
Extracts first 3 principal components and checks they correspond to the classic
level / slope / curvature interpretation.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca_on_yield_changes(yield_df, n_components=3):
    """
    yield_df: DataFrame, rows = dates, columns = maturities, values = yields (%)
    Returns dict with fitted PCA object, explained variance ratios, loadings, and PC scores.
    """
    changes = yield_df.diff().dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(changes.values)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T,
        index=yield_df.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    scores_df = pd.DataFrame(
        scores, index=changes.index, columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return {
        "pca": pca,
        "explained_variance_ratio": explained,
        "cumulative_variance": np.cumsum(explained),
        "loadings": loadings,
        "scores": scores_df,
    }


def interpret_components(loadings):
    """
    Heuristic check: PC1 loadings should be roughly constant-signed across maturities (level),
    PC2 should be monotonic (slope), PC3 should be a hump/U-shape (curvature).
    Returns a plain-text interpretation string per component.
    """
    interp = {}
    pc1 = loadings["PC1"].values
    pc2 = loadings["PC2"].values
    pc3 = loadings["PC3"].values if "PC3" in loadings.columns else None

    interp["PC1"] = "level (parallel shift)" if np.all(np.sign(pc1) == np.sign(pc1[0])) else "mixed"
    interp["PC2"] = "slope (short vs long end)" if (pc2[0] * pc2[-1] < 0) else "mixed"
    if pc3 is not None:
        mid = pc3[len(pc3) // 2]
        interp["PC3"] = "curvature (butterfly)" if (mid * pc3[0] < 0 and mid * pc3[-1] < 0) else "mixed"
    return interp


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_treasury_yields

    df = load_treasury_yields(use_live=True, start="2015-01-01", end="2024-12-31")
    result = run_pca_on_yield_changes(df)

    print("Explained variance ratio (PC1, PC2, PC3):", np.round(result["explained_variance_ratio"], 4))
    print("Cumulative variance explained by first 3 PCs: "
          f"{result['cumulative_variance'][-1]*100:.1f}%")
    print("\nLoadings:\n", result["loadings"])
    print("\nInterpretation:", interpret_components(result["loadings"]))
