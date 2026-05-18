from __future__ import annotations

from datetime import datetime, timedelta

from tradeagent.data.queries import (
    get_bars,
    latest_bar_ts,
    list_symbols,
    upsert_bars,
    upsert_instruments,
)


def _seed_bars(symbol: str = "AAPL", n: int = 30) -> list[dict]:
    base = datetime(2024, 1, 1)
    return [
        {
            "symbol": symbol,
            "ts": base + timedelta(days=i),
            "interval": "1d",
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "adj_close": 100.5 + i,
            "volume": 1_000_000.0,
        }
        for i in range(n)
    ]


def test_upsert_and_range_query():
    upsert_instruments([{"symbol": "AAPL", "name": "Apple", "exchange": "NASDAQ", "asset_type": "equity", "currency": "USD"}])
    rows = _seed_bars()
    n = upsert_bars(rows)
    assert n == 30

    df = get_bars("AAPL", start=datetime(2024, 1, 5), end=datetime(2024, 1, 15))
    assert len(df) == 11
    assert df["close"].iloc[0] == rows[4]["close"]


def test_latest_bar_ts_and_list_symbols():
    upsert_instruments([{"symbol": "AAPL", "name": "Apple", "exchange": "NASDAQ", "asset_type": "equity", "currency": "USD"}])
    upsert_bars(_seed_bars())
    assert "AAPL" in list_symbols()
    ts = latest_bar_ts("AAPL")
    assert ts is not None
    # last seed bar is day 29
    assert ts.day == 30 or ts.day == 29  # day arithmetic depending on driver
