"""Walk-forward evaluation and forecast metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone


def walk_forward(
    feats: pd.DataFrame, cols: list[str], model, min_train: int = 504, refit_every: int = 21
) -> pd.Series:
    """Expanding-window forecasts: refit every `refit_every` days on all data before the block.

    The target of row t is the return of day t+1, so the last training row of a block
    must be t-1 (its target, day t's return, is known at the close of day t).
    """
    X, y = feats[cols].to_numpy(), feats["target"].to_numpy()
    preds = np.full(len(feats), np.nan)
    for start in range(min_train, len(feats), refit_every):
        m = clone(model).fit(X[:start], y[:start])
        stop = min(start + refit_every, len(feats))
        preds[start:stop] = m.predict(X[start:stop])
    return pd.Series(preds, index=feats.index).dropna()


def metrics(pred: pd.Series, actual: pd.Series, benchmark: pd.Series) -> dict:
    """Hit rate, out-of-sample R^2 (vs `benchmark` forecasts) and Spearman IC."""
    a = actual.reindex(pred.index)
    moved = a != 0
    hit = float((np.sign(pred[moved]) == np.sign(a[moved])).mean())
    sse, sse_b = ((a - pred) ** 2).sum(), ((a - benchmark.reindex(pred.index)) ** 2).sum()
    ic = stats.spearmanr(pred, a).statistic if pred.std() > 0 else np.nan
    return {"hit_rate": hit, "r2_oos": float(1 - sse / sse_b), "ic": float(ic), "n": int(moved.sum())}


def hit_rate_pvalue(hits: int, n: int, p0: float) -> float:
    """One-sided binomial test that the hit rate beats p0 (e.g. the always-up rate)."""
    return float(stats.binomtest(hits, n, p0, alternative="greater").pvalue)
