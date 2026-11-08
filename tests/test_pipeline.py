import numpy as np
import pandas as pd
import pytest

from tweetsent import data, evaluation, features, models


@pytest.fixture(scope="module")
def aapl():
    return data.load("AAPL")


def test_holiday_calendar_matches_price_feed():
    """On every listed NYSE holiday and weekend the feed repeats the previous close."""
    cal = data.load_calendar_days("AAPL")
    prev = cal.close.shift(1)
    closed = cal.index[~data.is_trading_day(cal.index)][1:]  # first row has no previous close
    assert (cal.close[closed] == prev[closed]).all()


def test_weekend_tweets_fold_into_next_session(aapl):
    cal = data.load_calendar_days("AAPL")
    monday = pd.Timestamp("2015-01-05")  # Sat 3rd and Sun 4th fold into Mon 5th
    assert aapl.loc[monday, "n_days_folded"] == 3
    assert aapl.loc[monday, "trend_score"] == cal.loc["2015-01-03":"2015-01-05", "trend_score"].sum()
    assert data.is_trading_day(aapl.index).all()


def test_features_use_only_past_data(aapl):
    """Changing the future must not change today's features."""
    f1 = features.build(aapl)
    shocked = aapl.copy()
    cut = shocked.index[800]
    shocked.loc[shocked.index > cut, ["ret", "sentiment", "trend_score"]] *= 3.0
    f2 = features.build(shocked)
    cols = features.PRICE + features.SENTIMENT
    pd.testing.assert_frame_equal(f1.loc[:cut, cols], f2.loc[:cut, cols])


def test_target_is_next_day_return(aapl):
    f = features.build(aapl)
    nxt = aapl.ret.shift(-1).reindex(f.index)
    np.testing.assert_allclose(f.target, nxt)


def test_walk_forward_never_trains_on_the_test_block():
    """A model that memorises its training targets cannot score on unseen rows."""
    idx = pd.bdate_range("2015-01-01", periods=300)
    rng = np.random.default_rng(0)
    feats = pd.DataFrame({"x": rng.normal(size=300), "target": rng.normal(size=300)}, index=idx)

    class Lookup:
        def get_params(self, deep=True):
            return {}

        def fit(self, X, y):
            self.seen = set(map(float, X[:, 0]))
            return self

        def predict(self, X):
            assert not set(map(float, X[:, 0])) & self.seen
            return np.zeros(len(X))

    pred = evaluation.walk_forward(feats, ["x"], Lookup(), min_train=100, refit_every=50)
    assert len(pred) == 200


def test_oos_r2_is_zero_for_the_benchmark_itself():
    idx = pd.RangeIndex(10)
    a = pd.Series(np.arange(10.0), index=idx)
    b = pd.Series(4.5, index=idx)
    m = evaluation.metrics(b, a, b)
    assert m["r2_oos"] == 0.0


def test_models_fit_and_predict(aapl):
    f = features.build(aapl).iloc[:300]
    for m in models.make_models().values():
        p = m.fit(f[features.PRICE].to_numpy(), f.target.to_numpy()).predict(f[features.PRICE].to_numpy())
        assert p.shape == (300,) and np.isfinite(p).all()
