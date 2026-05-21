from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from tradeagent.data.db import connect
from tradeagent.data.models import forecasts
from tradeagent.data.queries import get_bars
from tradeagent.forecast.indicators import macd, rsi


@dataclass
class Forecast:
    symbol: str
    horizon_days: int
    point: float
    low: float
    high: float
    direction: str  # 'up' | 'down' | 'flat'
    r2_in_sample: float
    r2_walkforward: float
    last_close: float


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["vol_10"] = close.pct_change().rolling(10).std()
    out["rsi_14"] = rsi(close, 14)
    out["macd_hist"] = macd(close)["macd_hist"]
    dow = pd.get_dummies(df.index.dayofweek, prefix="dow").set_index(df.index)
    out = pd.concat([out, dow], axis=1)
    return out


def _walkforward_r2(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    n = len(X)
    if n < 60:
        return float("nan")
    fold_size = n // (n_splits + 1)
    scores: list[float] = []
    for k in range(1, n_splits + 1):
        train_end = fold_size * k
        test_end = min(train_end + fold_size, n)
        if test_end - train_end < 5:
            continue
        model = Ridge(alpha=1.0)
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
        pred = model.predict(X.iloc[train_end:test_end])
        scores.append(r2_score(y.iloc[train_end:test_end], pred))
    return float(np.mean(scores)) if scores else float("nan")


def forecast(
    symbol: str,
    horizon_days: int = 5,
    interval: str = "1d",
    persist: bool = True,
) -> Forecast:
    df = get_bars(symbol, interval=interval)
    if df.empty or len(df) < 80:
        raise ValueError(
            f"Only {len(df)} bars for {symbol}; need >= 80. Ingest more history "
            f"(`tradeagent ingest {symbol}`); a premium Alpha Vantage key unlocks full history."
        )

    feats = _build_features(df)
    y = df["close"].pct_change(horizon_days).shift(-horizon_days)
    data = feats.join(y.rename("target")).dropna()

    if len(data) < 60:
        raise ValueError(
            f"Insufficient feature rows for {symbol}: {len(data)} (need >= 60). "
            f"Ingest more history; a premium Alpha Vantage key unlocks full history."
        )

    X = data.drop(columns=["target"]).astype(float)
    y_clean = data["target"].astype(float)

    model = Ridge(alpha=1.0)
    model.fit(X, y_clean)
    r2_in = float(r2_score(y_clean, model.predict(X)))
    r2_wf = _walkforward_r2(X, y_clean)

    # Predict using latest feature row (must be re-computed without dropping last rows).
    latest_feats = feats.iloc[[-1]].astype(float)
    pred_return = float(model.predict(latest_feats)[0])
    last_close = float(df["close"].iloc[-1])
    point = last_close * (1.0 + pred_return)

    resid_std = float((y_clean - model.predict(X)).std())
    band = 1.96 * resid_std * last_close  # convert return std → price band

    if pred_return > 0.005:
        direction = "up"
    elif pred_return < -0.005:
        direction = "down"
    else:
        direction = "flat"

    fc = Forecast(
        symbol=symbol,
        horizon_days=horizon_days,
        point=point,
        low=point - band,
        high=point + band,
        direction=direction,
        r2_in_sample=r2_in,
        r2_walkforward=r2_wf,
        last_close=last_close,
    )

    if persist:
        confidence = max(0.0, min(1.0, r2_wf)) if not np.isnan(r2_wf) else 0.0
        with connect() as conn:
            conn.execute(
                forecasts.insert().values(
                    symbol=symbol,
                    made_at=datetime.utcnow(),
                    horizon_days=horizon_days,
                    model="ridge_v1",
                    predicted_close=point,
                    confidence=confidence,
                    rationale=str(asdict(fc)),
                )
            )
    return fc
