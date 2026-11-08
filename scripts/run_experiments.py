"""Reproduce every table and figure in the README.

    python scripts/run_experiments.py           # full run (~5 min on a laptop)
    python scripts/run_experiments.py --quick   # 2 stocks, fewer trees
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tweetsent import data, evaluation, features, models, replication, stats_tests  # noqa: E402

RESULTS = ROOT / "results"
REFIT_EVERY = 63  # trading days (one quarter)
warnings.filterwarnings("ignore")


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def run_replication(tickers) -> pd.DataFrame:
    rows = {tk: replication.run(data.load_calendar_days(tk)) for tk in tickers}
    return pd.DataFrame(rows).T


def run_link(tickers) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, quint = {}, []
    for tk in tickers:
        df = data.load(tk)
        rows[tk] = {**stats_tests.correlations(df), **{f"Granger p: {k}": v for k, v in stats_tests.granger(df).items()}}
        quint.append(stats_tests.quintile_returns(df))
    return pd.DataFrame(rows).T, sum(quint) / len(quint)


def run_forecasts(tickers, n_trees: int) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        feats = features.build(data.load(tk))
        bench = evaluation.walk_forward(feats, features.PRICE, models.HistoricalMean(), refit_every=REFIT_EVERY)
        base = evaluation.metrics(bench, feats.target, bench)
        rows.append({"ticker": tk, "features": "none", "model": "Historical mean (always up)", **base})
        for fs, cols in features.FEATURE_SETS.items():
            for name, m in models.make_models().items():
                if name == "Historical mean":
                    continue
                if name == "Random forest":
                    m.set_params(regressor__randomforestregressor__n_estimators=n_trees)
                pred = evaluation.walk_forward(feats, cols, m, refit_every=REFIT_EVERY)
                rows.append({"ticker": tk, "features": fs, "model": name, **evaluation.metrics(pred, feats.target, bench)})
        print(f"  {tk} done", flush=True)
    return pd.DataFrame(rows)


def fig_quintiles(q: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(q))
    ax.bar(x - 0.2, q["ret"], 0.4, label="Same day", color="#2b6cb0")
    ax.bar(x + 0.2, q["ret_next"], 0.4, label="Next day", color="#a0aec0")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [f"Q{i}" for i in q.index])
    ax.set_xlabel("Daily tweet-sentiment quintile (Q1 most negative)")
    ax.set_ylabel("Mean log return (bp)")
    ax.set_title("Sentiment moves with the price on the same day, not the next")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "returns_by_sentiment_quintile.png", dpi=150)
    plt.close(fig)


def fig_replication(rep: pd.DataFrame) -> None:
    cols = ["SVR, original scaling", "SVR, train-only scaling", "Yesterday's close (no model)"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(rep))
    for i, (c, col) in enumerate(zip(cols, ["#2b6cb0", "#e53e3e", "#38a169"])):
        ax.bar(x + (i - 1) * 0.27, rep[c].clip(lower=-1), 0.27, label=c, color=col)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, rep.index)
    ax.set_ylabel("Test R² on closing price (clipped at -1)")
    ax.set_title("The original 'accuracy' is price persistence")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULTS / "replication_r2.png", dpi=150)
    plt.close(fig)


def fig_hit_rates(fc: pd.DataFrame) -> None:
    piv = fc[fc.features == "price+sentiment"].pivot(index="model", columns="ticker", values="hit_rate")
    base = fc[fc.features == "none"].set_index("ticker").hit_rate
    excess = (piv - base) * 100
    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(excess, cmap="RdBu", vmin=-6, vmax=6)
    ax.set_xticks(range(excess.shape[1]), excess.columns)
    ax.set_yticks(range(excess.shape[0]), excess.index)
    for (i, j), v in np.ndenumerate(excess.to_numpy()):
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="pp vs always-up")
    ax.set_title("Next-day hit rate minus always-up baseline (price + sentiment features)")
    fig.tight_layout()
    fig.savefig(RESULTS / "hit_rate_vs_baseline.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    tickers = list(data.TICKERS)[:2] if args.quick else list(data.TICKERS)
    n_trees = 50 if args.quick else 300
    RESULTS.mkdir(exist_ok=True)

    print("1/3 replication of the original setup")
    rep = run_replication(tickers)
    rep.to_csv(RESULTS / "replication.csv")
    fig_replication(rep)

    print("2/3 sentiment-return link")
    link, quint = run_link(tickers)
    link.to_csv(RESULTS / "sentiment_return_link.csv")
    quint.to_csv(RESULTS / "returns_by_sentiment_quintile.csv")
    fig_quintiles(quint)

    print("3/3 walk-forward next-day forecasts")
    fc = run_forecasts(tickers, n_trees)
    fc.to_csv(RESULTS / "forecasts.csv", index=False)
    fig_hit_rates(fc)

    base = fc[fc.features == "none"].set_index("ticker")
    summary = (
        fc[fc.features != "none"]
        .assign(excess_hit=lambda d: d.hit_rate - d.ticker.map(base.hit_rate))
        .groupby(["features", "model"])[["hit_rate", "excess_hit", "r2_oos", "ic"]]
        .mean()
    )
    pooled = {}
    for (fs, m), g in fc[fc.features != "none"].groupby(["features", "model"]):
        hits = int(round((g.hit_rate * g.n).sum()))
        p0 = float((base.hit_rate * base.n).sum() / base.n.sum())
        pooled[(fs, m)] = evaluation.hit_rate_pvalue(hits, int(g.n.sum()), p0)
    summary["p_beats_always_up"] = pd.Series(pooled)
    summary.to_csv(RESULTS / "forecast_summary.csv")

    with open(RESULTS / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Results\n\n## Replication of the original setup (test R² on closing price)\n\n")
        f.write(md(rep) + "\n\n## Sentiment vs returns\n\n" + md(link, ".4f") + "\n\n")
        f.write("## Mean return by sentiment quintile (bp, averaged over stocks)\n\n" + md(quint, ".1f") + "\n\n")
        f.write("## Next-day forecasts, averaged over stocks\n\n" + md(summary, ".4f") + "\n\n")
        f.write("## Next-day forecasts, per stock\n\n" + md(fc.set_index(["ticker", "features", "model"]), ".4f") + "\n")
    print((RESULTS / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
