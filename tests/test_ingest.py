from __future__ import annotations

import httpx
import pytest

from tradeagent.data.ingest import AlphaVantageClient, ingest_symbol
from tradeagent.data.queries import get_bars


SAMPLE_DAILY = {
    "Meta Data": {"2. Symbol": "AAPL"},
    "Time Series (Daily)": {
        "2024-01-02": {
            "1. open": "100.0",
            "2. high": "101.0",
            "3. low": "99.0",
            "4. close": "100.5",
            "5. adjusted close": "100.5",
            "6. volume": "1000000",
        },
        "2024-01-03": {
            "1. open": "101.0",
            "2. high": "102.0",
            "3. low": "100.0",
            "4. close": "101.5",
            "5. adjusted close": "101.5",
            "6. volume": "1200000",
        },
    },
}


def test_alpha_vantage_fetch_daily(httpx_mock):
    httpx_mock.add_response(json=SAMPLE_DAILY)
    client = AlphaVantageClient()
    df = client.fetch_daily("AAPL")
    assert len(df) == 2
    assert set(df.columns) >= {"ts", "open", "close", "symbol", "interval"}
    assert df["symbol"].iloc[0] == "AAPL"


def test_ingest_symbol_writes_bars(httpx_mock):
    httpx_mock.add_response(json=SAMPLE_DAILY)
    n = ingest_symbol("AAPL", interval="1d", client=AlphaVantageClient())
    assert n == 2
    df = get_bars("AAPL")
    assert len(df) == 2


def test_throttle_response_raises_and_retries(httpx_mock):
    httpx_mock.add_response(json={"Note": "API rate limit"})
    httpx_mock.add_response(json={"Note": "API rate limit"})
    httpx_mock.add_response(json={"Note": "API rate limit"})
    httpx_mock.add_response(json={"Note": "API rate limit"})
    client = AlphaVantageClient()
    from tradeagent.data.ingest import AlphaVantageThrottled

    with pytest.raises(AlphaVantageThrottled):
        client.fetch_daily("AAPL")
