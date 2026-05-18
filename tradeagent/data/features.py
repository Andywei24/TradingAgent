from __future__ import annotations

import pandas as pd

from tradeagent.data.queries import get_bars, upsert_features
from tradeagent.forecast.indicators import CORE_FEATURES


def materialize_features(
    symbol: str,
    interval: str = "1d",
    feature_set: str = "core",
) -> int:
    if feature_set != "core":
        raise ValueError(f"unknown feature set: {feature_set}")

    df = get_bars(symbol, interval=interval)
    if df.empty:
        return 0

    rows: list[dict] = []
    for name, fn in CORE_FEATURES.items():
        series = fn(df).dropna()
        if isinstance(series, pd.DataFrame):
            # only single-column results expected here
            series = series.iloc[:, 0]
        for ts, value in series.items():
            rows.append(
                {
                    "symbol": symbol,
                    "ts": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    "interval": interval,
                    "feature_name": name,
                    "value": float(value),
                }
            )
    return upsert_features(rows)
