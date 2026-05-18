from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from tradeagent.data.queries import upsert_bars, upsert_instruments
from tradeagent.forecast.linear import forecast
from tradeagent.forecast.signals import decompose


def _seed_long_history(symbol: str = "TEST", n: int = 250) -> None:
    upsert_instruments([{"symbol": symbol, "name": symbol, "exchange": "NASDAQ", "asset_type": "equity", "currency": "USD"}])
    rng = np.random.default_rng(7)
    prices = 100 + rng.standard_normal(n).cumsum() * 0.5
    base = datetime(2024, 1, 1)
    rows = []
    for i in range(n):
        p = float(prices[i])
        rows.append(
            {
                "symbol": symbol,
                "ts": base + timedelta(days=i),
                "interval": "1d",
                "open": p,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p,
                "adj_close": p,
                "volume": 1_000_000.0,
            }
        )
    upsert_bars(rows)


def test_forecast_returns_valid_fields():
    _seed_long_history()
    fc = forecast("TEST", horizon_days=5, persist=False)
    assert fc.symbol == "TEST"
    assert fc.point > 0
    assert fc.low <= fc.point <= fc.high
    assert fc.direction in {"up", "down", "flat"}


def test_decompose_returns_trend_and_peaks():
    _seed_long_history(symbol="DCMP")
    from tradeagent.data.queries import get_bars

    close = get_bars("DCMP")["close"]
    res = decompose(close)
    assert len(res.trend) == len(close)
    assert isinstance(res.spectral_top_k, list)
