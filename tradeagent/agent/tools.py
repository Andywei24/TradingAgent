from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, Field

from tradeagent.data.queries import (
    get_bars,
    get_feature,
    list_symbols,
    summarize_statistics,
)
from tradeagent.forecast.indicators import CORE_FEATURES
from tradeagent.forecast.linear import forecast as run_forecast
from tradeagent.forecast.signals import decompose


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
    fc = run_forecast(inp.symbol, horizon_days=inp.horizon_days, interval=inp.interval)
    return asdict(fc)


def _tool_decompose_signal(inp: DecomposeSignalInput) -> dict:
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


def _tool_plot_chart(inp: PlotChartInput) -> dict:
    from tradeagent.viz.charts import plot_price_with_indicators

    path = plot_price_with_indicators(inp.symbol, indicators=inp.indicators, last_n=inp.last_n)
    return {"path": str(path)}


def _tool_summarize_statistics(inp: SummarizeStatisticsInput) -> dict:
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
        Tool("plot_chart", "Render a price chart with indicator overlays to PNG and return the path.", PlotChartInput, _tool_plot_chart),
        Tool("summarize_statistics", "Mean/std/skew/Sharpe over a recent window for a symbol.", SummarizeStatisticsInput, _tool_summarize_statistics),
    ]
}


def tool_schemas(names: list[str] | None = None) -> list[dict]:
    names = names or list(REGISTRY)
    return [REGISTRY[n].schema() for n in names if n in REGISTRY]
