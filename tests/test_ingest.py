from __future__ import annotations

import httpx
import pytest

from tradeagent.data.ingest import (
    AlphaVantageClient,
    AlphaVantagePremiumEndpoint,
    ingest_symbol,
)
from tradeagent.data.queries import get_bars


# Free-tier TIME_SERIES_DAILY shape: "5. volume", no adjusted close.
SAMPLE_DAILY = {
    "Meta Data": {"2. Symbol": "AAPL"},
    "Time Series (Daily)": {
        "2024-01-02": {
            "1. open": "100.0",
            "2. high": "101.0",
            "3. low": "99.0",
            "4. close": "100.5",
            "5. volume": "1000000",
        },
        "2024-01-03": {
            "1. open": "101.0",
            "2. high": "102.0",
            "3. low": "100.0",
            "4. close": "101.5",
            "5. volume": "1200000",
        },
    },
}


def test_alpha_vantage_fetch_daily(httpx_mock):
    httpx_mock.add_response(json=SAMPLE_DAILY)
    client = AlphaVantageClient()
    df = client.fetch_daily("AAPL")
    assert len(df) == 2
    assert set(df.columns) >= {"ts", "open", "close", "volume", "symbol", "interval"}
    assert df["symbol"].iloc[0] == "AAPL"
    # free tier has no adjusted close => adj_close falls back to close
    assert df["adj_close"].iloc[0] == df["close"].iloc[0]
    assert df["volume"].iloc[0] == 1000000.0


def test_uses_free_daily_endpoint_by_default(httpx_mock):
    # free tier => non-adjusted endpoint AND compact outputsize (full is premium-gated)
    httpx_mock.add_response(json=SAMPLE_DAILY)
    AlphaVantageClient().fetch_daily("AAPL", full=True)
    params = httpx_mock.get_requests()[0].url.params
    assert params["function"] == "TIME_SERIES_DAILY"
    assert params["outputsize"] == "compact"


def test_premium_endpoint_fails_fast_without_retry(httpx_mock):
    # A premium message must NOT be retried like a throttle.
    httpx_mock.add_response(
        json={"Information": "Thank you for using Alpha Vantage! This is a premium endpoint."}
    )
    client = AlphaVantageClient()
    with pytest.raises(AlphaVantagePremiumEndpoint):
        client.fetch_daily("AAPL")
    assert len(httpx_mock.get_requests()) == 1  # single call, no retry storm


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
