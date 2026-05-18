from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tradeagent.config import get_settings
from tradeagent.data.queries import get_bars, get_feature
from tradeagent.forecast.indicators import CORE_FEATURES


def _resolve_indicator(symbol: str, indicator: str, df: pd.DataFrame) -> pd.Series:
    series = get_feature(symbol, indicator)
    if not series.empty:
        return series
    fn = CORE_FEATURES.get(indicator)
    if fn is None:
        return pd.Series(dtype=float, name=indicator)
    out = fn(df)
    if isinstance(out, pd.DataFrame):
        out = out.iloc[:, 0]
    return out.rename(indicator)


def plot_price_with_indicators(
    symbol: str,
    indicators: list[str] | None = None,
    last_n: int = 180,
    out_dir: Path | None = None,
) -> Path:
    indicators = indicators or ["sma_20", "sma_50"]
    df = get_bars(symbol).tail(last_n)
    if df.empty:
        raise ValueError(f"no bars for {symbol}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["close"], label="close", linewidth=1.5)
    for name in indicators:
        series = _resolve_indicator(symbol, name, df).tail(last_n)
        if series.empty:
            continue
        ax.plot(series.index, series.values, label=name, alpha=0.8)
    ax.set_title(f"{symbol} — last {last_n} bars")
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()

    out_dir = out_dir or (Path(get_settings().data_dir) / "reports" / "charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{symbol}_{stamp}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_forecast(symbol: str, forecast, last_n: int = 60, out_dir: Path | None = None) -> Path:
    df = get_bars(symbol).tail(last_n)
    if df.empty:
        raise ValueError(f"no bars for {symbol}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["close"], label="close")
    # forecast point as a marker after the last date (offset by horizon)
    horizon = forecast.horizon_days
    last_ts = df.index[-1]
    future_ts = last_ts + pd.Timedelta(days=horizon)
    ax.scatter([future_ts], [forecast.point], color="orange", zorder=5, label="forecast")
    ax.fill_between(
        [last_ts, future_ts],
        [df["close"].iloc[-1], forecast.low],
        [df["close"].iloc[-1], forecast.high],
        color="orange",
        alpha=0.2,
        label="prediction band",
    )
    ax.set_title(
        f"{symbol} — {horizon}d forecast: {forecast.direction} "
        f"(R²_wf={forecast.r2_walkforward:.2f})"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()

    out_dir = out_dir or (Path(get_settings().data_dir) / "reports" / "charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{symbol}_forecast_{stamp}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
