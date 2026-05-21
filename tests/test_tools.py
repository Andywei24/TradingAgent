from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from tradeagent.agent.tools import REGISTRY
from tradeagent.data.queries import upsert_bars, upsert_instruments


def _seed_random_walk(symbol: str = "FCT", n: int = 250) -> None:
    upsert_instruments([{"symbol": symbol, "name": symbol, "exchange": "NASDAQ", "asset_type": "equity", "currency": "USD"}])
    rng = np.random.default_rng(11)
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


def test_run_linear_forecast_tool_generates_chart():
    _seed_random_walk()
    out = REGISTRY["run_linear_forecast"].call({"symbol": "FCT", "horizon_days": 5})
    assert "point" in out and "chart_path" in out
    p = Path(out["chart_path"])
    assert p.exists() and p.suffix == ".png"


def test_missing_symbol_returns_actionable_error():
    # No seeding: ZZZZ is absent from the store.
    for name, args in [
        ("get_price_history", {"symbol": "ZZZZ"}),
        ("summarize_statistics", {"symbol": "ZZZZ"}),
        ("compute_indicator", {"symbol": "ZZZZ", "indicator": "sma_20"}),
        ("run_linear_forecast", {"symbol": "ZZZZ"}),
        ("decompose_signal", {"symbol": "ZZZZ"}),
    ]:
        out = REGISTRY[name].call(args)
        assert out.get("missing_data") is True, name
        assert "ingest ZZZZ" in out["error"], name
        # no fabricated numbers leaked
        assert "rows" not in out and "point" not in out and "n" not in out, name


def test_auto_ingest_fetches_missing_symbol(httpx_mock):
    from tradeagent.agent.tools import auto_ingest_var

    sample = {
        "Meta Data": {"2. Symbol": "NEWS"},
        "Time Series (Daily)": {
            "2024-01-02": {"1. open": "10", "2. high": "11", "3. low": "9", "4. close": "10.5", "5. volume": "1000"},
            "2024-01-03": {"1. open": "10.5", "2. high": "11.5", "3. low": "10", "4. close": "11", "5. volume": "1200"},
        },
    }
    httpx_mock.add_response(json=sample)

    token = auto_ingest_var.set(True)
    try:
        out = REGISTRY["get_price_history"].call({"symbol": "NEWS", "last_n": 10})
    finally:
        auto_ingest_var.reset(token)

    assert "rows" in out and len(out["rows"]) == 2
    assert "missing_data" not in out
