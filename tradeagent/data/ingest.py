from __future__ import annotations

import csv
import io
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tradeagent.config import Settings, get_settings
from tradeagent.data.db import connect
from tradeagent.data.models import data_ingest_log
from tradeagent.data.queries import latest_bar_ts, upsert_bars, upsert_instruments

log = logging.getLogger(__name__)


class AlphaVantageThrottled(RuntimeError):
    """Raised when Alpha Vantage returns a Note/Information rate-limit response (retryable)."""


class AlphaVantagePremiumEndpoint(RuntimeError):
    """Raised when an endpoint requires a premium subscription (NOT retryable)."""


class MarketDataClient(ABC):
    @abstractmethod
    def fetch_daily(self, symbol: str, full: bool = False) -> pd.DataFrame: ...

    @abstractmethod
    def fetch_intraday(self, symbol: str, interval: str = "60min") -> pd.DataFrame: ...

    @abstractmethod
    def list_nasdaq_symbols(self) -> list[dict]: ...


@dataclass
class _TokenBucket:
    """Simple per-minute token bucket. Blocks until a slot is available."""

    rate_per_min: int
    _stamps: deque = None

    def __post_init__(self):
        self._stamps = deque(maxlen=self.rate_per_min)

    def acquire(self) -> None:
        now = time.monotonic()
        if len(self._stamps) == self.rate_per_min:
            elapsed = now - self._stamps[0]
            if elapsed < 60:
                time.sleep(60 - elapsed + 0.05)
                now = time.monotonic()
        self._stamps.append(now)


class AlphaVantageClient(MarketDataClient):
    """Thin wrapper around the Alpha Vantage REST API with rate limiting + retries."""

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or get_settings()
        self.client = client or httpx.Client(timeout=30.0)
        self.bucket = _TokenBucket(rate_per_min=self.settings.alphavantage_rate_limit_per_min)

    # ----- low-level -----
    def _get(self, params: dict) -> httpx.Response:
        params = {**params, "apikey": self.settings.alphavantage_api_key}
        self.bucket.acquire()
        t0 = time.monotonic()
        resp = self.client.get(self.settings.alphavantage_base_url, params=params)
        latency = int((time.monotonic() - t0) * 1000)
        resp.raise_for_status()
        # log every call regardless of success
        try:
            with connect() as conn:
                conn.execute(
                    data_ingest_log.insert().values(
                        symbol=params.get("symbol"),
                        endpoint=params.get("function"),
                        status=str(resp.status_code),
                        bytes=len(resp.content),
                        latency_ms=latency,
                    )
                )
        except Exception as e:  # logging must not break ingest
            log.warning("ingest log write failed: %s", e)
        return resp

    def _get_json(self, params: dict) -> dict:
        for attempt in Retrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            retry=retry_if_exception_type(AlphaVantageThrottled),
            reraise=True,
        ):
            with attempt:
                resp = self._get(params)
                data = resp.json()
                note = data.get("Note") or data.get("Information")
                if note:
                    # "premium endpoint" / "premium API function" => won't ever succeed
                    # on a free key, so fail fast instead of burning retries.
                    if "premium" in note.lower():
                        log.error("Alpha Vantage premium-only endpoint: %s", note)
                        raise AlphaVantagePremiumEndpoint(note)
                    log.warning("Alpha Vantage throttle: %s", note)
                    raise AlphaVantageThrottled(note)
                if "Error Message" in data:
                    raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
                return data
        raise RuntimeError("unreachable")  # pragma: no cover

    # ----- public -----
    def fetch_daily(self, symbol: str, full: bool = False) -> pd.DataFrame:
        # On the free tier, both DAILY_ADJUSTED and outputsize=full are premium-gated.
        premium = self.settings.alphavantage_premium
        function = "TIME_SERIES_DAILY_ADJUSTED" if premium else "TIME_SERIES_DAILY"
        outputsize = "full" if (full and premium) else "compact"
        params = {
            "function": function,
            "symbol": symbol,
            "outputsize": outputsize,
            "datatype": "json",
        }
        data = self._get_json(params)
        series = data.get("Time Series (Daily)")
        if not series:
            return pd.DataFrame()
        df = _av_series_to_frame(series, daily=True)
        df["symbol"] = symbol
        df["interval"] = "1d"
        return df

    def fetch_intraday(self, symbol: str, interval: str = "60min") -> pd.DataFrame:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "full" if self.settings.alphavantage_premium else "compact",
            "datatype": "json",
        }
        data = self._get_json(params)
        key = f"Time Series ({interval})"
        series = data.get(key)
        if not series:
            return pd.DataFrame()
        df = _av_series_to_frame(series, daily=False)
        df["symbol"] = symbol
        df["interval"] = interval
        return df

    def list_nasdaq_symbols(self) -> list[dict]:
        params = {"function": "LISTING_STATUS", "state": "active"}
        self.bucket.acquire()
        resp = self.client.get(self.settings.alphavantage_base_url, params={**params, "apikey": self.settings.alphavantage_api_key})
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        ex = self.settings.alphavantage_default_exchange
        return [
            {
                "symbol": row["symbol"],
                "name": row.get("name") or "",
                "exchange": row.get("exchange") or "",
                "asset_type": row.get("assetType") or "equity",
                "currency": "USD",
            }
            for row in reader
            if row.get("exchange") == ex
        ]


def _av_series_to_frame(series: dict, daily: bool) -> pd.DataFrame:
    records = []
    for ts_str, row in series.items():
        records.append(
            {
                "ts": datetime.fromisoformat(ts_str) if "T" in ts_str or len(ts_str) > 10 else datetime.strptime(ts_str, "%Y-%m-%d"),
                "open": float(row.get("1. open", 0.0)),
                "high": float(row.get("2. high", 0.0)),
                "low": float(row.get("3. low", 0.0)),
                "close": float(row.get("4. close", 0.0)),
                "adj_close": float(row.get("5. adjusted close", row.get("4. close", 0.0))) if daily else float(row.get("4. close", 0.0)),
                "volume": float(row.get("6. volume", row.get("5. volume", 0.0))),
            }
        )
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values("ts").reset_index(drop=True)
    return df


class CSVAdapter(MarketDataClient):
    """Offline fallback: reads data/csv/{symbol}.csv with columns ts,open,high,low,close,volume."""

    def __init__(self, root: Path):
        self.root = root

    def fetch_daily(self, symbol: str, full: bool = False) -> pd.DataFrame:
        path = self.root / f"{symbol}.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, parse_dates=["ts"])
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        df["symbol"] = symbol
        df["interval"] = "1d"
        return df

    def fetch_intraday(self, symbol: str, interval: str = "60min") -> pd.DataFrame:
        return pd.DataFrame()

    def list_nasdaq_symbols(self) -> list[dict]:
        rows = []
        for path in sorted(self.root.glob("*.csv")):
            rows.append(
                {
                    "symbol": path.stem,
                    "name": path.stem,
                    "exchange": "NASDAQ",
                    "asset_type": "equity",
                    "currency": "USD",
                }
            )
        return rows


def make_client(settings: Settings | None = None) -> MarketDataClient:
    settings = settings or get_settings()
    if settings.alphavantage_api_key:
        return AlphaVantageClient(settings)
    log.warning("ALPHAVANTAGE_API_KEY not set — using CSV fallback at %s/csv", settings.data_dir)
    return CSVAdapter(Path(settings.data_dir) / "csv")


def _rows_for_upsert(df: pd.DataFrame) -> list[dict]:
    cols = ["symbol", "ts", "interval", "open", "high", "low", "close", "adj_close", "volume"]
    return df[cols].to_dict(orient="records")


def ingest_symbol(
    symbol: str,
    interval: str = "1d",
    full: bool = False,
    client: MarketDataClient | None = None,
) -> int:
    client = client or make_client()
    # Ensure instrument row exists before bars (FK).
    upsert_instruments([{"symbol": symbol, "name": symbol, "exchange": "NASDAQ", "asset_type": "equity", "currency": "USD"}])

    last_ts = latest_bar_ts(symbol, interval)
    use_full = full or last_ts is None or (datetime.utcnow() - last_ts).days > 90

    if interval == "1d":
        df = client.fetch_daily(symbol, full=use_full)
    else:
        df = client.fetch_intraday(symbol, interval=interval)

    if df.empty:
        return 0
    if last_ts is not None:
        df = df[df["ts"] > last_ts]
    if df.empty:
        return 0
    return upsert_bars(_rows_for_upsert(df))


def ingest_many(symbols: Iterable[str], interval: str = "1d", full: bool = False) -> dict[str, int]:
    client = make_client()
    out: dict[str, int] = {}
    for sym in symbols:
        try:
            n = ingest_symbol(sym, interval=interval, full=full, client=client)
            out[sym] = n
            log.info("ingested %s: %d new bars", sym, n)
        except (
            RetryError,
            AlphaVantageThrottled,
            AlphaVantagePremiumEndpoint,
            httpx.HTTPError,
            ValueError,
        ) as e:
            log.error("ingest failed for %s: %s", sym, e)
            out[sym] = 0
    return out


def seed_nasdaq_universe(limit: int | None = None) -> int:
    client = make_client()
    rows = client.list_nasdaq_symbols()
    if limit:
        rows = rows[:limit]
    return upsert_instruments(rows)
