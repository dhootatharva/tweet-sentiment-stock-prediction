"""Build data/processed/daily_<company>.csv from the raw tweet dataset.

Raw data: "Tweets about the Top Companies from 2015 to 2020" on Kaggle
(files Tweet.csv, Company_Tweet.csv, Company.csv, CompanyValues.csv). Download it
yourself (Kaggle account needed), unzip into data/raw/, then run

    pip install -e ".[prepare]"
    python scripts/prepare_data.py --raw data/raw

Steps: score every tweet with VADER (compound score), drop neutral tweets (score 0),
compute a trend score per tweet (1 + comments + retweets + likes), join tweets to the
daily price table by date and ticker, then per company and day keep the close, the
summed trend score and the mean sentiment. This takes roughly 20-30 minutes (~4M tweets).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"AAPL": "apple", "AMZN": "amazon", "GOOG": "google", "GOOGL": "google", "MSFT": "microsoft", "TSLA": "tesla"}


def vader_scores(texts: pd.Series) -> pd.Series:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk.download("vader_lexicon", quiet=True)
    sid = SentimentIntensityAnalyzer()
    return texts.map(lambda t: sid.polarity_scores(str(t))["compound"])


def build(raw: Path, out: Path) -> None:
    tweets = pd.read_csv(raw / "Tweet.csv")
    links = pd.read_csv(raw / "Company_Tweet.csv")
    prices = pd.read_csv(raw / "CompanyValues.csv", parse_dates=["day_date"])

    tweets = tweets.merge(links, on="tweet_id")
    tweets["date"] = pd.to_datetime(tweets.post_date, unit="s").dt.normalize()
    tweets["sentiment"] = vader_scores(tweets.body)
    tweets["trend_score"] = tweets.comment_num + tweets.retweet_num + tweets.like_num + 1
    tweets = tweets[tweets.sentiment != 0]

    prices = prices[prices.day_date < "2020-01-01"].rename(columns={"day_date": "date"})
    merged = tweets.merge(prices[["ticker_symbol", "date", "close_value"]], on=["ticker_symbol", "date"])
    merged["company"] = merged.ticker_symbol.map(NAMES)

    out.mkdir(parents=True, exist_ok=True)
    for company, g in merged.groupby("company"):
        trend = g.groupby(["date", "close_value"], as_index=False).trend_score.sum()
        sent = g.groupby("date", as_index=False).sentiment.mean()
        daily = trend.merge(sent, on="date").rename(columns={"date": "date_str_x"})
        daily.to_csv(out / f"daily_{company}.csv")
        print(f"{company}: {len(daily)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    a = ap.parse_args()
    build(a.raw, a.out)
