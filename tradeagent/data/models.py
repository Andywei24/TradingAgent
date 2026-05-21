from __future__ import annotations

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    TIMESTAMP,
    func,
)

metadata = MetaData()

instruments = Table(
    "instruments",
    metadata,
    Column("symbol", String, primary_key=True),
    Column("name", String),
    Column("exchange", String),
    Column("asset_type", String),
    Column("currency", String),
    Column("created_at", TIMESTAMP, server_default=func.current_timestamp()),
)

ohlcv_bars = Table(
    "ohlcv_bars",
    metadata,
    Column("symbol", String, ForeignKey("instruments.symbol"), nullable=False),
    Column("ts", TIMESTAMP, nullable=False),
    Column("interval", String, nullable=False),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("adj_close", Float),
    Column("volume", Float),
    PrimaryKeyConstraint("symbol", "interval", "ts"),
    Index("idx_ohlcv_symbol_ts", "symbol", "ts"),
)

features = Table(
    "features",
    metadata,
    Column("symbol", String, nullable=False),
    Column("ts", TIMESTAMP, nullable=False),
    Column("interval", String, nullable=False),
    Column("feature_name", String, nullable=False),
    Column("value", Float),
    PrimaryKeyConstraint("symbol", "interval", "ts", "feature_name"),
    Index("idx_features_lookup", "symbol", "feature_name", "ts"),
)

forecasts = Table(
    "forecasts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String),
    Column("made_at", TIMESTAMP),
    Column("horizon_days", Integer),
    Column("model", String),
    Column("predicted_close", Float),
    Column("confidence", Float),
    Column("rationale", Text),
)

agent_runs = Table(
    "agent_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("started_at", TIMESTAMP),
    Column("user_query", Text),
    Column("final_answer", Text),
    Column("tool_trace", Text),
    Column("artifacts", Text),
)

data_ingest_log = Table(
    "data_ingest_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", TIMESTAMP, server_default=func.current_timestamp()),
    Column("symbol", String),
    Column("endpoint", String),
    Column("status", String),
    Column("bytes", Integer),
    Column("latency_ms", Integer),
    Column("note", Text),
)
