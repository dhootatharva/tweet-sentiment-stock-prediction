"""Does sentiment lead returns, or follow them?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests


def correlations(df: pd.DataFrame) -> dict:
    """Pearson correlation of daily sentiment with same-day and next-day returns."""
    d = df.dropna(subset=["ret", "ret_next", "sentiment"])
    same = stats.pearsonr(d.sentiment, d.ret)
    nxt = stats.pearsonr(d.sentiment, d.ret_next)
    return {"corr_same_day": same.statistic, "p_same_day": same.pvalue,
            "corr_next_day": nxt.statistic, "p_next_day": nxt.pvalue}


def granger(df: pd.DataFrame, maxlag: int = 5) -> dict:
    """Smallest SSR F-test p-value over lags 1..maxlag, in both directions."""
    d = df[["ret", "sentiment"]].dropna()

    def min_p(cols: list[str]) -> float:
        res = grangercausalitytests(d[cols].to_numpy(), maxlag=maxlag)
        return min(r[0]["ssr_ftest"][1] for r in res.values())

    return {"sentiment -> returns": min_p(["ret", "sentiment"]), "returns -> sentiment": min_p(["sentiment", "ret"])}


def quintile_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Mean same-day and next-day return (bp) by sentiment quintile (quintiles per stock)."""
    d = df.dropna(subset=["ret", "ret_next", "sentiment"]).copy()
    d["q"] = pd.qcut(d.sentiment, 5, labels=[1, 2, 3, 4, 5])
    return d.groupby("q", observed=True)[["ret", "ret_next"]].mean() * 1e4
