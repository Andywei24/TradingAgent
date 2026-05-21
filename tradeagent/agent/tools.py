from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable
from urllib.parse import urlparse

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from tradeagent.config import get_settings
from tradeagent.data.queries import (
    get_bars,
    get_feature,
    latest_bar_ts,
    list_symbols,
    summarize_statistics,
)
from tradeagent.forecast.indicators import CORE_FEATURES
from tradeagent.forecast.linear import forecast as run_forecast
from tradeagent.forecast.signals import decompose

log = logging.getLogger(__name__)

# Request-scoped: when True, symbol-scoped tools auto-ingest a missing symbol instead
# of failing. Set by run_chain so the tool functions themselves stay stateless.
auto_ingest_var: ContextVar[bool] = ContextVar("auto_ingest", default=False)


def _ensure_symbol_data(symbol: str, interval: str = "1d") -> dict | None:
    """Guard against operating on a symbol that has no bars in the local store.

    Returns None when data is present (proceed). Otherwise returns an error dict the
    tool should hand straight back to the model — either a clear "run ingest" hint, or,
    when auto-ingest is enabled, the result of a best-effort on-the-fly fetch.
    """
    if latest_bar_ts(symbol, interval) is not None:
        return None

    if auto_ingest_var.get():
        from tradeagent.data.features import materialize_features
        from tradeagent.data.ingest import ingest_symbol

        try:
            n = ingest_symbol(symbol, interval=interval)
        except Exception as e:  # throttle / premium / network
            log.warning("auto-ingest failed for %s: %s", symbol, e)
            return {
                "error": f"Auto-ingest failed for {symbol}: {type(e).__name__}: {e}",
                "fetch_failed": True,
            }
        if n == 0 or latest_bar_ts(symbol, interval) is None:
            return {
                "error": f"No data returned for {symbol} from the data provider.",
                "fetch_failed": True,
            }
        try:
            materialize_features(symbol, interval=interval)
        except Exception as e:  # features are best-effort; bars are enough to proceed
            log.warning("feature build failed for %s after auto-ingest: %s", symbol, e)
        return None

    return {
        "error": f"No data for {symbol} in the local store. Run: tradeagent ingest {symbol}",
        "missing_data": True,
    }


# --------- input schemas ---------


class ListSymbolsInput(BaseModel):
    asset_type: str | None = Field(default=None, description="equity | etf | crypto | fx")


class GetPriceHistoryInput(BaseModel):
    symbol: str
    interval: str = Field(default="1d")
    start: str | None = Field(default=None, description="ISO date, inclusive")
    end: str | None = Field(default=None, description="ISO date, inclusive")
    last_n: int | None = Field(default=120, description="If set, return only the last N bars")


class ComputeIndicatorInput(BaseModel):
    symbol: str
    indicator: str = Field(description="One of CORE_FEATURES, e.g. sma_20, rsi_14, macd_hist")
    interval: str = Field(default="1d")
    last_n: int = Field(default=60)


class RunLinearForecastInput(BaseModel):
    symbol: str
    horizon_days: int = Field(default=5, ge=1, le=30)
    interval: str = Field(default="1d")


class DecomposeSignalInput(BaseModel):
    symbol: str
    interval: str = Field(default="1d")
    last_n: int = Field(default=180)


class RetrieveKnowledgeInput(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)


class SearchMarketNewsInput(BaseModel):
    query: str = Field(description="Natural-language news query")
    symbol: str | None = Field(default=None, description="Optional ticker symbol to focus the search")
    days: int | None = Field(default=None, ge=1, le=30)
    max_results: int | None = Field(default=None, ge=1, le=10)


class PlotChartInput(BaseModel):
    symbol: str
    indicators: list[str] = Field(default_factory=lambda: ["sma_20", "sma_50"])
    last_n: int = Field(default=180)


class SummarizeStatisticsInput(BaseModel):
    symbol: str
    interval: str = Field(default="1d")
    window_days: int = Field(default=60, ge=5, le=2520)


# --------- tool implementations ---------


def _tool_list_symbols(inp: ListSymbolsInput) -> dict:
    return {"symbols": list_symbols(asset_type=inp.asset_type)}


def _slice_last_n(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    return df.tail(n) if n else df


def _tool_get_price_history(inp: GetPriceHistoryInput) -> dict:
    guard = _ensure_symbol_data(inp.symbol, inp.interval)
    if guard is not None:
        return guard
    start = datetime.fromisoformat(inp.start) if inp.start else None
    end = datetime.fromisoformat(inp.end) if inp.end else None
    df = get_bars(inp.symbol, interval=inp.interval, start=start, end=end)
    df = _slice_last_n(df, inp.last_n)
    if df.empty:
        return {"symbol": inp.symbol, "rows": []}
    payload = df.reset_index()[["ts", "open", "high", "low", "close", "volume"]].copy()
    payload["ts"] = payload["ts"].astype(str)
    return {"symbol": inp.symbol, "rows": payload.to_dict(orient="records")}


def _tool_compute_indicator(inp: ComputeIndicatorInput) -> dict:
    if inp.indicator not in CORE_FEATURES:
        return {"error": f"unknown indicator '{inp.indicator}'", "available": list(CORE_FEATURES)}
    guard = _ensure_symbol_data(inp.symbol, inp.interval)
    if guard is not None:
        return guard
    # Try cached features first; fall back to live compute.
    series = get_feature(inp.symbol, inp.indicator, interval=inp.interval)
    if series.empty:
        df = get_bars(inp.symbol, interval=inp.interval)
        if df.empty:
            return {"error": f"no bars for {inp.symbol}"}
        out = CORE_FEATURES[inp.indicator](df)
        if isinstance(out, pd.DataFrame):
            out = out.iloc[:, 0]
        series = out.dropna()
    series = series.tail(inp.last_n)
    return {
        "symbol": inp.symbol,
        "indicator": inp.indicator,
        "values": [
            {"ts": str(ts), "value": float(v)} for ts, v in series.items()
        ],
    }


def _tool_run_linear_forecast(inp: RunLinearForecastInput) -> dict:
    guard = _ensure_symbol_data(inp.symbol, inp.interval)
    if guard is not None:
        return guard
    try:
        fc = run_forecast(inp.symbol, horizon_days=inp.horizon_days, interval=inp.interval)
    except ValueError as e:  # symbol exists but not enough history
        return {"error": str(e), "insufficient_data": True}
    result = asdict(fc)
    # Visualize the forecast right after running it; a plotting failure must not break
    # the forecast itself.
    try:
        from tradeagent.viz.charts import plot_forecast

        path = plot_forecast(inp.symbol, fc)
        result["chart_path"] = str(path)
    except Exception as e:  # pragma: no cover - defensive
        import logging

        logging.getLogger(__name__).warning("forecast chart failed for %s: %s", inp.symbol, e)
    return result


def _tool_decompose_signal(inp: DecomposeSignalInput) -> dict:
    guard = _ensure_symbol_data(inp.symbol, inp.interval)
    if guard is not None:
        return guard
    df = get_bars(inp.symbol, interval=inp.interval)
    if df.empty:
        return {"error": f"no bars for {inp.symbol}"}
    close = df["close"].tail(inp.last_n)
    res = decompose(close)
    return {
        "symbol": inp.symbol,
        "dominant_period_days": res.dominant_period_days,
        "top_peaks": [{"period_days": p, "amplitude": a} for p, a in res.spectral_top_k],
        "trend_slope_per_day": float(
            (res.trend.iloc[-1] - res.trend.iloc[0]) / max(1, len(res.trend) - 1)
        )
        if len(res.trend) > 1
        else 0.0,
    }


def _tool_retrieve_knowledge(inp: RetrieveKnowledgeInput) -> dict:
    from tradeagent.rag.retriever import retrieve

    hits = retrieve(inp.query, k=inp.k)
    return {"query": inp.query, "hits": hits}


def _source_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _tool_search_market_news(inp: SearchMarketNewsInput) -> dict:
    settings = get_settings()
    if not settings.tavily_api_key:
        return {
            "error": "TAVILY_API_KEY is not configured; set it in .env to enable market news search.",
            "news_unavailable": True,
        }

    days = inp.days or settings.tavily_news_days
    max_results = inp.max_results or settings.tavily_max_results
    query = inp.query.strip()
    if inp.symbol:
        query = f"{inp.symbol.upper()} stock market news {query}"

    payload = {
        "query": query,
        "topic": "news",
        "days": days,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
    }
    headers = {"Authorization": f"Bearer {settings.tavily_api_key}"}

    try:
        resp = httpx.post(settings.tavily_base_url, json=payload, headers=headers, timeout=20.0)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        return {
            "error": f"Tavily news search failed: {type(e).__name__}: {e}",
            "news_unavailable": True,
        }

    results = []
    for item in data.get("results", [])[:max_results]:
        url = str(item.get("url") or "")
        if not url:
            continue
        results.append(
            {
                "title": item.get("title") or "",
                "url": url,
                "source": _source_from_url(url),
                "published_date": item.get("published_date") or item.get("publishedDate"),
                "snippet": item.get("content") or "",
                "score": item.get("score"),
            }
        )

    return {
        "query": data.get("query") or query,
        "symbol": inp.symbol,
        "days": days,
        "results": results,
    }


def _tool_plot_chart(inp: PlotChartInput) -> dict:
    guard = _ensure_symbol_data(inp.symbol)
    if guard is not None:
        return guard
    from tradeagent.viz.charts import plot_price_with_indicators

    path = plot_price_with_indicators(inp.symbol, indicators=inp.indicators, last_n=inp.last_n)
    return {"path": str(path)}


def _tool_summarize_statistics(inp: SummarizeStatisticsInput) -> dict:
    guard = _ensure_symbol_data(inp.symbol, inp.interval)
    if guard is not None:
        return guard
    return summarize_statistics(inp.symbol, interval=inp.interval, window_days=inp.window_days)


# --------- registry ---------


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        fn: Callable[[BaseModel], dict],
    ):
        self.name = name
        self.description = description
        self.input_model = input_model
        self.fn = fn

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }

    def call(self, raw_args: str | dict) -> dict:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        validated = self.input_model.model_validate(args)
        return self.fn(validated)


REGISTRY: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("list_symbols", "List symbols available in the local market-data store.", ListSymbolsInput, _tool_list_symbols),
        Tool("get_price_history", "Fetch OHLCV bars for a symbol from the local SQL store.", GetPriceHistoryInput, _tool_get_price_history),
        Tool("compute_indicator", "Compute or fetch a cached technical indicator series.", ComputeIndicatorInput, _tool_compute_indicator),
        Tool("run_linear_forecast", "Fit a ridge regression on engineered features and predict the horizon return; returns point, interval, direction, walk-forward R².", RunLinearForecastInput, _tool_run_linear_forecast),
        Tool("decompose_signal", "FFT-based spectral decomposition of recent prices; returns dominant period and trend slope.", DecomposeSignalInput, _tool_decompose_signal),
        Tool("retrieve_knowledge", "Retrieve top-k snippets from the financial knowledge RAG store.", RetrieveKnowledgeInput, _tool_retrieve_knowledge),
        Tool("search_market_news", "Search recent market/news web results for a ticker or financial question using Tavily; returns titles, snippets, sources, URLs, and dates.", SearchMarketNewsInput, _tool_search_market_news),
        Tool("plot_chart", "Render a price chart with indicator overlays to PNG and return the path.", PlotChartInput, _tool_plot_chart),
        Tool("summarize_statistics", "Mean/std/skew/Sharpe over a recent window for a symbol.", SummarizeStatisticsInput, _tool_summarize_statistics),
    ]
}


def tool_schemas(names: list[str] | None = None) -> list[dict]:
    names = names or list(REGISTRY)
    return [REGISTRY[n].schema() for n in names if n in REGISTRY]
