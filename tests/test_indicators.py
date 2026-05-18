from __future__ import annotations

import numpy as np
import pandas as pd

from tradeagent.forecast.indicators import bollinger, ema, macd, rsi, sma


def _series(n: int = 100, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 + rng.standard_normal(n).cumsum(), name="close")


def test_sma_matches_pandas_rolling():
    s = _series()
    expected = s.rolling(20).mean()
    out = sma(s, 20)
    pd.testing.assert_series_equal(out.dropna(), expected.dropna(), check_names=False)


def test_ema_is_finite_and_length():
    s = _series()
    out = ema(s, 12)
    assert len(out) == len(s)
    assert np.isfinite(out.iloc[-1])


def test_rsi_bounded_0_100():
    s = _series(200)
    out = rsi(s, 14).dropna()
    assert (out >= 0).all() and (out <= 100).all()


def test_macd_columns_present():
    s = _series()
    df = macd(s)
    assert {"macd", "macd_signal", "macd_hist"} <= set(df.columns)
    # macd_hist = macd - signal
    np.testing.assert_allclose(
        (df["macd"] - df["macd_signal"]).dropna().values,
        df["macd_hist"].dropna().values,
    )


def test_bollinger_band_ordering():
    s = _series()
    bb = bollinger(s, 20).dropna()
    assert (bb["boll_upper"] >= bb["boll_mid"]).all()
    assert (bb["boll_mid"] >= bb["boll_lower"]).all()
