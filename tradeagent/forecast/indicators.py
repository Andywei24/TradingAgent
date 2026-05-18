from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int = 20) -> pd.Series:
    return close.rolling(window=window, min_periods=window).mean().rename(f"sma_{window}")


def ema(close: pd.Series, window: int = 20) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean().rename(f"ema_{window}")


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / window, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.rename(f"rsi_{window}")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
        }
    )


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    return pd.DataFrame(
        {
            "boll_mid": mid,
            "boll_upper": mid + k * std,
            "boll_lower": mid - k * std,
        }
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean().rename(f"atr_{window}")


CORE_FEATURES: dict[str, callable] = {
    "sma_20": lambda df: sma(df["close"], 20),
    "sma_50": lambda df: sma(df["close"], 50),
    "ema_12": lambda df: ema(df["close"], 12),
    "ema_26": lambda df: ema(df["close"], 26),
    "rsi_14": lambda df: rsi(df["close"], 14),
    "macd": lambda df: macd(df["close"])["macd"],
    "macd_signal": lambda df: macd(df["close"])["macd_signal"],
    "macd_hist": lambda df: macd(df["close"])["macd_hist"],
    "boll_upper": lambda df: bollinger(df["close"])["boll_upper"],
    "boll_lower": lambda df: bollinger(df["close"])["boll_lower"],
    "atr_14": lambda df: atr(df["high"], df["low"], df["close"], 14),
}
