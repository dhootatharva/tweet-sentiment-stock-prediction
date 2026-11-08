"""Models: the SVR and random forest from the original study, plus linear and naive baselines.

Every model is a scikit-learn pipeline whose scalers are fit on training data only.
Targets (daily returns, ~1e-2) are standardised so the SVR's epsilon-tube is meaningful.
"""
from __future__ import annotations

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np


class HistoricalMean(BaseEstimator, RegressorMixin):
    """Forecast = mean training return (the standard benchmark for out-of-sample R^2)."""

    def fit(self, X, y):
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)


def _scaled(estimator) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(make_pipeline(StandardScaler(), estimator), transformer=StandardScaler())


def make_models(seed: int = 0) -> dict:
    return {
        "Historical mean": HistoricalMean(),
        "Ridge": _scaled(Ridge(alpha=10.0)),
        "SVR (RBF)": _scaled(SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        "Random forest": _scaled(
            RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=20, n_jobs=-1, random_state=seed)
        ),
    }
