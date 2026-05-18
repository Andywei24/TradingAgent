from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import detrend


@dataclass
class SignalDecomposition:
    trend: pd.Series
    residual: pd.Series
    dominant_period_days: float | None
    spectral_top_k: list[tuple[float, float]]  # (period_days, amplitude)


def decompose(close: pd.Series, top_k: int = 3) -> SignalDecomposition:
    series = close.dropna().astype(float)
    if len(series) < 8:
        return SignalDecomposition(
            trend=pd.Series(index=series.index, dtype=float),
            residual=pd.Series(index=series.index, dtype=float),
            dominant_period_days=None,
            spectral_top_k=[],
        )
    x = series.values
    trend = x - detrend(x, type="linear")
    residual = x - trend

    spectrum = np.fft.rfft(residual - residual.mean())
    freqs = np.fft.rfftfreq(len(residual), d=1.0)
    amps = np.abs(spectrum)
    # ignore DC bin
    if len(amps) <= 1:
        peaks = []
        dominant = None
    else:
        amps[0] = 0.0
        order = np.argsort(amps)[::-1][:top_k]
        peaks = [
            (float(1.0 / freqs[i]) if freqs[i] > 0 else float("inf"), float(amps[i]))
            for i in order
        ]
        dominant = peaks[0][0] if peaks else None

    return SignalDecomposition(
        trend=pd.Series(trend, index=series.index, name="trend"),
        residual=pd.Series(residual, index=series.index, name="residual"),
        dominant_period_days=dominant,
        spectral_top_k=peaks,
    )


def zscore_anomalies(close: pd.Series, window: int = 30, threshold: float = 2.5) -> pd.Series:
    returns = close.pct_change()
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    z = (returns - mean) / std
    return (z.abs() > threshold).fillna(False).rename("anomaly")
