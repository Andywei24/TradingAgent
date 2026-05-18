from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
from sqlalchemy import and_, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from tradeagent.data.db import connect
from tradeagent.data.models import features, instruments, ohlcv_bars


def upsert_instruments(rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with connect() as conn:
        stmt = sqlite_insert(instruments).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "name": stmt.excluded.name,
                "exchange": stmt.excluded.exchange,
                "asset_type": stmt.excluded.asset_type,
                "currency": stmt.excluded.currency,
            },
        )
        conn.execute(stmt)
    return len(rows)


def upsert_bars(rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with connect() as conn:
        stmt = sqlite_insert(ohlcv_bars).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "interval", "ts"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "adj_close": stmt.excluded.adj_close,
                "volume": stmt.excluded.volume,
            },
        )
        conn.execute(stmt)
    return len(rows)


def upsert_features(rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with connect() as conn:
        stmt = sqlite_insert(features).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "interval", "ts", "feature_name"],
            set_={"value": stmt.excluded.value},
        )
        conn.execute(stmt)
    return len(rows)


def get_bars(
    symbol: str,
    interval: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    conditions = [ohlcv_bars.c.symbol == symbol, ohlcv_bars.c.interval == interval]
    if start is not None:
        conditions.append(ohlcv_bars.c.ts >= start)
    if end is not None:
        conditions.append(ohlcv_bars.c.ts <= end)
    stmt = select(ohlcv_bars).where(and_(*conditions)).order_by(ohlcv_bars.c.ts.asc())
    with connect() as conn:
        df = pd.read_sql(stmt, conn)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
    return df


def latest_bar_ts(symbol: str, interval: str = "1d") -> datetime | None:
    stmt = select(func.max(ohlcv_bars.c.ts)).where(
        and_(ohlcv_bars.c.symbol == symbol, ohlcv_bars.c.interval == interval)
    )
    with connect() as conn:
        row = conn.execute(stmt).scalar_one_or_none()
    if isinstance(row, str):
        return datetime.fromisoformat(row)
    return row


def get_feature(
    symbol: str,
    feature_name: str,
    interval: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.Series:
    conditions = [
        features.c.symbol == symbol,
        features.c.feature_name == feature_name,
        features.c.interval == interval,
    ]
    if start is not None:
        conditions.append(features.c.ts >= start)
    if end is not None:
        conditions.append(features.c.ts <= end)
    stmt = (
        select(features.c.ts, features.c.value)
        .where(and_(*conditions))
        .order_by(features.c.ts.asc())
    )
    with connect() as conn:
        df = pd.read_sql(stmt, conn)
    if df.empty:
        return pd.Series(dtype=float, name=feature_name)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")["value"].rename(feature_name)


def list_symbols(asset_type: str | None = None) -> list[str]:
    stmt = select(instruments.c.symbol).order_by(instruments.c.symbol.asc())
    if asset_type is not None:
        stmt = stmt.where(instruments.c.asset_type == asset_type)
    with connect() as conn:
        return [row[0] for row in conn.execute(stmt).all()]


def summarize_statistics(
    symbol: str,
    interval: str = "1d",
    window_days: int = 60,
) -> dict:
    end = datetime.utcnow()
    df = get_bars(symbol, interval=interval)
    if df.empty:
        return {"symbol": symbol, "n": 0}
    tail = df.tail(window_days)
    rets = tail["close"].pct_change().dropna()
    stats = {
        "symbol": symbol,
        "n": int(len(tail)),
        "mean_close": float(tail["close"].mean()),
        "last_close": float(tail["close"].iloc[-1]),
        "return_mean": float(rets.mean()),
        "return_std": float(rets.std()),
        "return_skew": float(rets.skew()) if len(rets) > 2 else 0.0,
        "sharpe_annualized": float(rets.mean() / rets.std() * (252**0.5))
        if rets.std() > 0
        else 0.0,
        "window_days": window_days,
        "as_of": end.isoformat(),
    }
    return stats
