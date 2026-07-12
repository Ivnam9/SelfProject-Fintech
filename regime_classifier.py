"""
Regime Detection
-----------------
Classifies each day into an interest-rate regime using:
  (a) a Hidden Markov Model (hmmlearn.GaussianHMM) on [slope, level, 3M rate-of-change], and
  (b) a simple threshold-based rule set, as a transparent sanity check against (a).

Features:
  - slope  = 10Y - 2Y yield
  - level  = 10Y yield
  - 3M ROC = 63-trading-day change in 10Y yield (rate of change)
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def build_features(yield_df):
    slope = yield_df["10Y"] - yield_df["2Y"]
    level = yield_df["10Y"]
    roc_3m = yield_df["10Y"].diff(63)
    feats = pd.DataFrame({"slope": slope, "level": level, "roc_3m": roc_3m}).dropna()
    return feats


def fit_hmm_regimes(features, n_states=4, seed=42):
    X = features.values
    model = GaussianHMM(n_components=n_states, covariance_type="full",
                         n_iter=200, random_state=seed)
    model.fit(X)
    hidden_states = model.predict(X)
    return model, hidden_states


def label_states(features, hidden_states, n_states=4):
    """
    Assigns a human-readable label to each HMM state based on the mean
    slope/level/roc within that state, so labels are data-driven rather
    than hardcoded to a fixed state index.
    """
    df = features.copy()
    df["state"] = hidden_states
    state_means = df.groupby("state")[["slope", "level", "roc_3m"]].mean()

    labels = {}
    for state, row in state_means.iterrows():
        if row["slope"] < 0:
            labels[state] = "Inverted"
        elif row["roc_3m"] > 0.15:
            labels[state] = "Rising Rates"
        elif row["roc_3m"] < -0.15:
            labels[state] = "Falling Rates"
        else:
            labels[state] = "Flat / Range-bound"
    return labels


def threshold_classify(features):
    """Simple, transparent rule-based classifier for cross-checking the HMM output."""
    def rule(row):
        if row["slope"] < 0:
            return "Inverted"
        elif row["roc_3m"] > 0.15:
            return "Rising Rates"
        elif row["roc_3m"] < -0.15:
            return "Falling Rates"
        else:
            return "Flat / Range-bound"
    return features.apply(rule, axis=1)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_treasury_yields

    df = load_treasury_yields(use_live=True, start="2015-01-01", end="2024-12-31")
    feats = build_features(df)

    model, states = fit_hmm_regimes(feats, n_states=4)
    labels = label_states(feats, states, n_states=4)

    feats = feats.copy()
    feats["hmm_state"] = states
    feats["hmm_regime"] = feats["hmm_state"].map(labels)
    feats["rule_regime"] = threshold_classify(feats)

    agreement = (feats["hmm_regime"] == feats["rule_regime"]).mean()
    print(f"HMM vs rule-based agreement: {agreement*100:.1f}%")
    print("\nRegime distribution (HMM):")
    print(feats["hmm_regime"].value_counts())
    print("\nSample:\n", feats.tail(10)[["slope", "level", "roc_3m", "hmm_regime", "rule_regime"]])
