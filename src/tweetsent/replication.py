"""Re-run of the original 2022 setup, and the two changes that make it an honest test.

Original setup (per stock): calendar-day rows; features = close, sentiment and trend
score over the previous 3 days; target = today's close; first 70% train, last 30% test;
MinMax scalers fit separately on the train AND the test set; score = R^2 on scaled prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


def window_data(df: pd.DataFrame, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    v = df[["close", "sentiment", "trend_score"]].to_numpy()
    X = np.array([np.concatenate([v[i:i + window, j] for j in range(3)]) for i in range(len(v) - window)])
    y = v[window:, 0]
    return X, y


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    X, y = window_data(df)
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split].reshape(-1, 1), y[split:].reshape(-1, 1)

    def fit_score(model, train_scalers_only: bool) -> float:
        xs, ys = MinMaxScaler().fit(Xtr), MinMaxScaler().fit(ytr)
        xs_te = xs if train_scalers_only else MinMaxScaler().fit(Xte)
        ys_te = ys if train_scalers_only else MinMaxScaler().fit(yte)
        model.fit(xs.transform(Xtr), ys.transform(ytr).ravel())
        pred = ys_te.inverse_transform(model.predict(xs_te.transform(Xte)).reshape(-1, 1))
        return r2_score(yte, pred)

    rf = lambda: RandomForestRegressor(n_estimators=1000, max_depth=2, bootstrap=False, random_state=seed, n_jobs=-1)
    naive = X[split:, 2]  # yesterday's close (last of the 3 lagged closes)
    return {
        "SVR, original scaling": fit_score(svm.SVR(), False),
        "Random forest, original scaling": fit_score(rf(), False),
        "SVR, train-only scaling": fit_score(svm.SVR(), True),
        "Random forest, train-only scaling": fit_score(rf(), True),
        "Yesterday's close (no model)": r2_score(yte.ravel(), naive),
    }
