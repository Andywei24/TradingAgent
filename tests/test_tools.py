from __future__ import annotations

from datetime import datetime, timedelta

from tradeagent.agent.tools import REGISTRY
from tradeagent.data.queries import upsert_bars, upsert_instruments


def _seed(symbol: str = "AAPL", n: int = 60) -> None:
    upsert_instruments([{"symbol": symbol, "name": symbol, "exchange": "NASDAQ", "asset_type": "equity", "currency": "USD"}])
    base = datetime(2024, 1, 1)
    rows = [
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
    upsert_bars(rows)


def test_list_symbols_tool():
    _seed()
    out = REGISTRY["list_symbols"].call({})
    assert "AAPL" in out["symbols"]


def test_get_price_history_tool_returns_rows():
    _seed()
    out = REGISTRY["get_price_history"].call({"symbol": "AAPL", "last_n": 10})
    assert len(out["rows"]) == 10
    assert "close" in out["rows"][0]


def test_compute_indicator_falls_back_to_live():
    _seed(n=80)
    out = REGISTRY["compute_indicator"].call({"symbol": "AAPL", "indicator": "sma_20", "last_n": 20})
    assert "values" in out and len(out["values"]) == 20


def test_summarize_statistics_tool():
    _seed(n=80)
    out = REGISTRY["summarize_statistics"].call({"symbol": "AAPL", "window_days": 30})
    assert out["n"] == 30
    assert "sharpe_annualized" in out


def test_unknown_indicator_returns_error():
    _seed()
    out = REGISTRY["compute_indicator"].call({"symbol": "AAPL", "indicator": "bogus", "last_n": 5})
    assert "error" in out
