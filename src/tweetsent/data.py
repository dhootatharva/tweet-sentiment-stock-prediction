"""Load the daily tweet-sentiment panels and align them to NYSE trading days.

Each processed file has one row per calendar day (2015-01-01 to 2019-12-31):
the closing price, the summed trend score (1 + comments + retweets + likes over
all non-neutral tweets that day) and the mean VADER compound score of those tweets.
The price feed carries the last close forward over weekends and holidays, so
non-trading days are folded into the next trading day: their tweets are
information that arrives before that day's open.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
TICKERS = {"AAPL": "apple", "AMZN": "amazon", "GOOG": "google", "MSFT": "microsoft", "TSLA": "tesla"}

# NYSE full-day closures inside the sample (weekends are handled separately).
NYSE_HOLIDAYS = pd.to_datetime([
    "2015-01-01", "2015-01-19", "2015-02-16", "2015-04-03", "2015-05-25", "2015-07-03",
    "2015-09-07", "2015-11-26", "2015-12-25",
    "2016-01-01", "2016-01-18", "2016-02-15", "2016-03-25", "2016-05-30", "2016-07-04",
    "2016-09-05", "2016-11-24", "2016-12-26",
    "2017-01-02", "2017-01-16", "2017-02-20", "2017-04-14", "2017-05-29", "2017-07-04",
    "2017-09-04", "2017-11-23", "2017-12-25",
    "2018-01-01", "2018-01-15", "2018-02-19", "2018-03-30", "2018-05-28", "2018-07-04",
    "2018-09-03", "2018-11-22", "2018-12-05", "2018-12-25",
    "2019-01-01", "2019-01-21", "2019-02-18", "2019-04-19", "2019-05-27", "2019-07-04",
    "2019-09-02", "2019-11-28", "2019-12-25",
])


def is_trading_day(dates: pd.DatetimeIndex) -> np.ndarray:
    return (dates.dayofweek < 5) & ~dates.isin(NYSE_HOLIDAYS)


def load_calendar_days(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per calendar day: close, trend_score, sentiment.

    Google appears under two share classes (GOOG/GOOGL) with two closes per day;
    their closes are averaged and trend scores summed.
    """
    raw = pd.read_csv(data_dir / f"daily_{TICKERS[ticker]}.csv", index_col=0, parse_dates=["date_str_x"])
    return (
        raw.groupby("date_str_x")
        .agg(close=("close_value", "mean"), trend_score=("trend_score", "sum"), sentiment=("sentiment", "mean"))
        .rename_axis("date")
    )


def to_trading_days(cal: pd.DataFrame) -> pd.DataFrame:
    """Fold weekend/holiday tweets into the next trading day.

    Sentiment of the folded block is the trend-score-weighted mean of its daily
    means (weights approximate tweet activity); trend scores are summed.
    """
    trading = is_trading_day(cal.index)
    # every calendar day maps to the next trading day on or after it
    session = pd.Series(cal.index.where(trading), index=cal.index).bfill()
    df = cal.assign(session=session.values, w_sent=cal.sentiment * cal.trend_score).dropna(subset=["session"])
    out = df.groupby("session").agg(trend_score=("trend_score", "sum"), w_sent=("w_sent", "sum"))
    out["sentiment"] = out.w_sent / out.trend_score
    out["close"] = cal.close.reindex(out.index)
    out["n_days_folded"] = df.groupby("session").size()
    return out.drop(columns="w_sent").rename_axis("date")[["close", "trend_score", "sentiment", "n_days_folded"]]


def load(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Trading-day panel with next-day log return as the prediction target."""
    df = to_trading_days(load_calendar_days(ticker, data_dir))
    df["ret"] = np.log(df.close).diff()
    df["ret_next"] = df.ret.shift(-1)
    return df
