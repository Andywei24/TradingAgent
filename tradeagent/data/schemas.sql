CREATE TABLE IF NOT EXISTS instruments (
    symbol     TEXT PRIMARY KEY,
    name       TEXT,
    exchange   TEXT,
    asset_type TEXT,
    currency   TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ohlcv_bars (
    symbol     TEXT NOT NULL,
    ts         TIMESTAMP NOT NULL,
    interval   TEXT NOT NULL,
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    adj_close  REAL,
    volume     REAL,
    PRIMARY KEY (symbol, interval, ts)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_ts ON ohlcv_bars(symbol, ts DESC);

CREATE TABLE IF NOT EXISTS features (
    symbol       TEXT NOT NULL,
    ts           TIMESTAMP NOT NULL,
    interval     TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value        REAL,
    PRIMARY KEY (symbol, interval, ts, feature_name)
);
CREATE INDEX IF NOT EXISTS idx_features_lookup ON features(symbol, feature_name, ts DESC);

CREATE TABLE IF NOT EXISTS forecasts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT,
    made_at         TIMESTAMP,
    horizon_days    INTEGER,
    model           TEXT,
    predicted_close REAL,
    confidence      REAL,
    rationale       TEXT
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TIMESTAMP,
    user_query   TEXT,
    final_answer TEXT,
    tool_trace   TEXT
);

CREATE TABLE IF NOT EXISTS data_ingest_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    symbol    TEXT,
    endpoint  TEXT,
    status    TEXT,
    bytes     INTEGER,
    latency_ms INTEGER,
    note      TEXT
);
