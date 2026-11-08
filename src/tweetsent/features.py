"""Features known at the close of day t, used to predict the return of day t+1."""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE = ["ret_0", "ret_1", "ret_2", "ret_3", "ret_4", "mom_5", "vol_5"]
SENTIMENT = ["sent_0", "sent_1", "sent_2", "sent_chg", "sent_z20", "log_attn", "attn_z20"]
FEATURE_SETS = {"price": PRICE, "sentiment": SENTIMENT, "price+sentiment": PRICE + SENTIMENT}


def _z(x: pd.Series, window: int) -> pd.Series:
    """Deviation from the trailing mean in trailing standard deviations (past data only)."""
    m, s = x.rolling(window).mean(), x.rolling(window).std()
    return (x - m) / s


def build(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for k in range(5):
        f[f"ret_{k}"] = df.ret.shift(k)
    f["mom_5"] = df.ret.rolling(5).sum()
    f["vol_5"] = df.ret.rolling(5).std()
    for k in range(3):
        f[f"sent_{k}"] = df.sentiment.shift(k)
    f["sent_chg"] = df.sentiment.diff()
    f["sent_z20"] = _z(df.sentiment, 20)
    f["log_attn"] = np.log1p(df.trend_score)
    f["attn_z20"] = _z(f.log_attn, 20)
    f["target"] = df.ret_next
    return f.dropna()
