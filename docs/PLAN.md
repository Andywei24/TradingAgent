# Trading Agent — Implementation Plan

## 1. Goals

Build a reasoning-driven trading research assistant that can:

1. Store and serve historical OHLCV + engineered feature data from a SQL backend tuned for fast time-range queries.
2. Use a Deepseek-V3 LLM as the planner/executor, with a tool registry and prompt-chained workflow (plan → select tool → execute → synthesize).
3. Forecast short-horizon price movement via linear regression and classical signal-processing indicators.
4. Communicate findings with statistical summaries and chart artifacts.
5. Ground answers in a curated financial-knowledge corpus through RAG.

Out of scope (for v1): live order execution, real broker integration, multi-user serving, intraday streaming.

---

## 2. High-Level Architecture

```
                         ┌────────────────────────┐
       user prompt ─────▶│  Reasoning Agent (LLM) │◀─── prompt chain
                         │   Deepseek-V3 core     │
                         └─────────┬──────────────┘
                                   │ tool calls
            ┌──────────────────────┼──────────────────────────┐
            ▼                      ▼                          ▼
   ┌────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
   │ Market Data SQL│   │ Forecast / Signals  │   │ RAG Knowledge Store │
   │ (SQLite/Postgres│  │ (sklearn, scipy,    │   │ (FAISS/Chroma +     │
   │  + indexes)    │   │  statsmodels)       │   │  embeddings)        │
   └────────┬───────┘   └──────────┬──────────┘   └──────────┬──────────┘
            │                      │                          │
            └──────────┬───────────┴──────────┬───────────────┘
                       ▼                      ▼
                ┌─────────────┐        ┌─────────────┐
                │ Visualizer  │        │   Report    │
                │ (matplotlib │───────▶│ (Markdown + │
                │  / plotly)  │        │  PNG bundle)│
                └─────────────┘        └─────────────┘
```

---

## 3. Tech Stack

| Layer            | Choice                              | Rationale |
|------------------|-------------------------------------|-----------|
| Language         | Python 3.11+                        | ecosystem for finance / ML / LLM |
| DB (dev)         | SQLite (WAL mode)                   | zero-config, fast for single-user |
| DB (prod-ready)  | PostgreSQL + TimescaleDB (optional) | hypertables for time-series scale |
| ORM / driver     | SQLAlchemy 2.x Core                 | portable across SQLite/Postgres |
| Data ingest      | **Alpha Vantage REST API** (NASDAQ symbols), CSV adapter | official feed; documented endpoints; free tier covers daily bars |
| ML / stats       | scikit-learn, statsmodels, scipy    | linear regression, signal processing |
| LLM              | Deepseek-V3 via OpenAI-compatible API | tool-use, JSON mode |
| Embeddings       | `sentence-transformers` (BGE-small) | local, free; swap to Deepseek embed |
| Vector store     | FAISS (local file)                  | fast, no server |
| Viz              | matplotlib + mplfinance, plotly opt | candles, indicators |
| CLI              | Typer                               | ergonomic |
| Config           | pydantic-settings + `.env`          | typed config |
| Tests            | pytest                              | |

---

## 4. Repository Layout

```
TradingAgent/
├── pyproject.toml
├── .env.example
├── README.md
├── docs/
│   └── PLAN.md                  # this file
├── tradeagent/
│   ├── __init__.py
│   ├── config.py
│   ├── cli.py
│   ├── data/
│   │   ├── db.py                # engine, session, migrations
│   │   ├── schemas.sql          # canonical DDL
│   │   ├── models.py            # SQLAlchemy table defs
│   │   ├── ingest.py            # fetch + upsert OHLCV (Alpha Vantage client)
│   │   ├── features.py          # materialize feature rows
│   │   └── queries.py           # typed query helpers (used by agent tools)
│   ├── forecast/
│   │   ├── indicators.py        # SMA, EMA, RSI, MACD, Bollinger
│   │   ├── signals.py           # FFT, detrending, change-points
│   │   └── linear.py            # OLS / ridge regression forecaster
│   ├── agent/
│   │   ├── deepseek_client.py   # thin wrapper around chat-completions
│   │   ├── prompts.py           # system + chain templates
│   │   ├── tools.py             # JSON-schema tool registry
│   │   └── chain.py             # planner→executor→synthesizer loop
│   ├── rag/
│   │   ├── loader.py            # PDF / HTML / md ingestion
│   │   ├── embedder.py          # batch embed + cache
│   │   ├── vectorstore.py       # FAISS wrapper
│   │   └── retriever.py         # top-k + reranker, exposed as tool
│   └── viz/
│       ├── charts.py            # price + indicator overlays
│       └── report.py            # bundle markdown report
├── tests/
│   ├── test_db.py
│   ├── test_indicators.py
│   ├── test_forecast.py
│   ├── test_tools.py
│   └── test_agent_chain.py
└── data/                        # gitignored
    ├── market.db
    ├── faiss_index/
    └── knowledge_base/          # raw source docs
```

---

## 5. Component Detail

### 5.1 SQL Market Data Store (`tradeagent/data/`)

**Tables**

```sql
-- instruments: one row per tradable symbol
CREATE TABLE instruments (
    symbol     TEXT PRIMARY KEY,
    name       TEXT,
    exchange   TEXT,
    asset_type TEXT,           -- equity, etf, crypto, fx
    currency   TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ohlcv_bars: canonical price history
CREATE TABLE ohlcv_bars (
    symbol     TEXT NOT NULL,
    ts         TIMESTAMP NOT NULL,
    interval   TEXT NOT NULL,   -- '1d','1h','5m'
    open       REAL, high REAL, low REAL, close REAL,
    adj_close  REAL,
    volume     REAL,
    PRIMARY KEY (symbol, interval, ts),
    FOREIGN KEY (symbol) REFERENCES instruments(symbol)
);
CREATE INDEX idx_ohlcv_symbol_ts ON ohlcv_bars(symbol, ts DESC);

-- features: long-format engineered features
CREATE TABLE features (
    symbol       TEXT NOT NULL,
    ts           TIMESTAMP NOT NULL,
    interval     TEXT NOT NULL,
    feature_name TEXT NOT NULL,    -- 'sma_20','rsi_14','macd_signal'
    value        REAL,
    PRIMARY KEY (symbol, interval, ts, feature_name)
);
CREATE INDEX idx_features_lookup ON features(symbol, feature_name, ts DESC);

-- forecasts: agent-generated predictions, kept for evaluation
CREATE TABLE forecasts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT,
    made_at         TIMESTAMP,
    horizon_days    INTEGER,
    model           TEXT,
    predicted_close REAL,
    confidence      REAL,
    rationale       TEXT
);

-- agent_runs: trace of LLM sessions for debugging / eval
CREATE TABLE agent_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TIMESTAMP,
    user_query  TEXT,
    final_answer TEXT,
    tool_trace  TEXT             -- JSON
);
```

**Query helpers** (`queries.py`) expose narrow, typed accessors so the agent never writes free-form SQL:

- `get_bars(symbol, interval, start, end) -> DataFrame`
- `latest_bar(symbol, interval) -> Bar`
- `get_feature(symbol, feature_name, start, end) -> Series`
- `list_symbols(asset_type=None) -> list[str]`

**Performance notes**

- SQLite: enable WAL + `synchronous=NORMAL`, `mmap_size=256MB`.
- Composite PK on `(symbol, interval, ts)` covers the dominant access path.
- Materialize indicators once into `features` rather than recomputing per request.
- Hook for swap to TimescaleDB: same DDL minus `AUTOINCREMENT`, plus `create_hypertable('ohlcv_bars','ts')`.

**Market data source — Alpha Vantage**

NASDAQ market data is fetched from the [Alpha Vantage](https://www.alphavantage.co/) REST API. The ingest module (`tradeagent/data/ingest.py`) wraps the HTTP client and handles auth, pagination, rate limits, and upsert.

- **Auth**: API key in `ALPHAVANTAGE_API_KEY` env var; passed as `&apikey=...` query string.
- **Base URL**: `https://www.alphavantage.co/query`.
- **Endpoints used**:
  - `TIME_SERIES_DAILY_ADJUSTED` — daily OHLCV + adjusted close + dividend/split (primary feed for v1).
  - `TIME_SERIES_INTRADAY` (`interval=5min|15min|60min`, `outputsize=full`, `month=YYYY-MM`) — intraday bars when needed.
  - `LISTING_STATUS` — bulk CSV of active NASDAQ/NYSE tickers; used to seed `instruments`.
  - `OVERVIEW` — company fundamentals, written into `instruments.name`/`asset_type` and an optional `fundamentals` table later.
- **Symbol filter**: `LISTING_STATUS` rows where `exchange = 'NASDAQ'` define the universe for v1; the CLI accepts an explicit symbol list to override.
- **Response shape**: JSON; parse `"Time Series (Daily)"` dict → DataFrame → upsert into `ohlcv_bars` with `interval='1d'`.
- **Rate limiting**: free tier is 25 requests/day, 5/min. The client must:
  - read `ALPHAVANTAGE_RATE_LIMIT_PER_MIN` (default 5) and throttle with a token bucket,
  - detect the `"Note"` / `"Information"` keys Alpha Vantage returns on throttle and back off (exponential, capped at 60s),
  - persist a `data_ingest_log` row per call (symbol, endpoint, status, bytes, latency) so re-runs are idempotent.
- **Incremental loads**: query `MAX(ts) WHERE symbol=? AND interval=?` and only request bars after that date (`outputsize=compact` if gap < 100 bars, else `full`).
- **Fallback / dev**: if `ALPHAVANTAGE_API_KEY` is unset, the ingest module falls back to a `CSVAdapter` that loads from `data/csv/{symbol}.csv` so the rest of the pipeline can be developed offline.
- **Upgrade path**: the `MarketDataClient` is an interface (`fetch_daily`, `fetch_intraday`, `list_symbols`); a Polygon/Tiingo implementation can be dropped in without touching downstream modules.

### 5.2 Forecast & Signal Processing (`tradeagent/forecast/`)

- **`indicators.py`** — pure functions over a pandas `Series`/`DataFrame`: `sma`, `ema`, `rsi`, `macd`, `bollinger`, `atr`. Vectorized; no I/O.
- **`signals.py`** — detrending (`scipy.signal.detrend`), spectral decomposition (`numpy.fft.rfft`), simple change-point flag, z-score anomaly tag.
- **`linear.py`** — `LinearForecaster`:
  - features: lagged returns (1,5,10), rolling vol, RSI, MACD hist, day-of-week one-hot
  - model: `sklearn.linear_model.Ridge` (small alpha) — robust to multicollinearity
  - output: `Forecast(point=..., low=..., high=..., direction='up'|'down'|'flat', r2_in_sample=..., r2_walkforward=...)`
  - persists `Forecast` row to DB

### 5.3 Reasoning Agent (`tradeagent/agent/`)

**Deepseek client** — OpenAI-SDK-compatible. Reads `DEEPSEEK_API_KEY`, base URL `https://api.deepseek.com`. Supports function-calling / `tools=[...]`.

**Tool registry** (initial set, JSON-schema described):

| Tool name              | Purpose |
|------------------------|---------|
| `list_symbols`         | what's in the DB |
| `get_price_history`    | OHLCV for symbol + range |
| `compute_indicator`    | run any indicator from `forecast.indicators` |
| `run_linear_forecast`  | fit + predict horizon |
| `decompose_signal`     | FFT / trend / residual |
| `retrieve_knowledge`   | RAG query → top-k snippets |
| `plot_chart`           | save PNG, return path |
| `summarize_statistics` | mean/std/skew/sharpe over a window |

Each tool is a thin Python function with a Pydantic input model; the registry auto-builds the JSON schema from the model.

**Prompt chain** (`chain.py`)

1. **Planner prompt** — receives user query, returns a JSON plan: list of sub-goals + which tools each likely needs. No tool calls yet.
2. **Executor loop** — for each sub-goal, run a tool-calling chat turn with a focused system prompt and the relevant tool subset. Repeat until the model emits no tool call.
3. **Synthesizer prompt** — given the plan + tool outputs + retrieved knowledge, produce the final analyst-style answer (with cited snippets and chart paths).

Every step is logged to `agent_runs.tool_trace` for replay.

### 5.4 RAG — Financial Knowledge (`tradeagent/rag/`)

- **Corpus** (seed): Investopedia articles (allowed scrape or manual export), classic texts in public domain (Graham, Fama papers), CFA Level-1 outline notes, SEC EDGAR primer, glossary of derivatives.
- **Loader** — supports `.pdf` (pypdf), `.md`, `.html` (trafilatura). Chunk: 800-token windows with 100-token overlap, attach `{source, title, section}` metadata.
- **Embedder** — `BAAI/bge-small-en-v1.5` via `sentence-transformers` (local CPU is fine for v1).
- **Vector store** — FAISS `IndexFlatIP` for v1; switch to `IndexHNSWFlat` if corpus > 50k chunks.
- **Retriever** — `retrieve_knowledge(query, k=5)` returns chunks + scores + source metadata. Optional cross-encoder rerank (`bge-reranker-base`).
- Exposed as the `retrieve_knowledge` tool to the agent.

### 5.5 Visualization & Reporting (`tradeagent/viz/`)

- `plot_price_with_indicators(symbol, indicators=[...], window=...)` → PNG, returns path.
- `plot_forecast(symbol, forecast)` → fan chart with prediction interval.
- `build_report(run_id)` → assembles a markdown file embedding charts + agent answer + cited knowledge snippets, saved under `data/reports/`.

### 5.6 CLI (`tradeagent/cli.py`)

```
tradeagent ingest AAPL MSFT --interval 1d --since 2018-01-01    # Alpha Vantage TIME_SERIES_DAILY_ADJUSTED
tradeagent ingest --universe nasdaq-top50 --interval 1d         # bulk seed via LISTING_STATUS
tradeagent features build --symbols all --set core
tradeagent rag index ./data/knowledge_base
tradeagent ask "Compare momentum on NVDA vs AMD over the last 6 months and forecast next week."
tradeagent report 42                       # rebuild report for agent_run id 42
```

---

## 6. Data & Control Flow (Example Query)

User: *"Is NVDA overbought right now, and what's a sensible 5-day forecast?"*

1. CLI → `agent.chain.run(query)`.
2. **Planner** returns plan: `[fetch_recent_prices, compute_rsi, run_forecast, retrieve_knowledge('overbought RSI interpretation'), synthesize]`.
3. **Executor** calls tools sequentially; each tool reads/writes the SQL store.
4. **Synthesizer** drafts: RSI value + interpretation (citing RAG snippet), forecast point + interval, chart path.
5. `agent_runs` row written. `viz.report.build_report` produces `data/reports/run_42.md`.

---

## 7. Configuration (`.env.example`)

```
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat              # V3

# Market data — Alpha Vantage (NASDAQ feed)
ALPHAVANTAGE_API_KEY=...
ALPHAVANTAGE_BASE_URL=https://www.alphavantage.co/query
ALPHAVANTAGE_RATE_LIMIT_PER_MIN=5         # free tier: 5/min, 25/day
ALPHAVANTAGE_DEFAULT_EXCHANGE=NASDAQ

DB_URL=sqlite:///data/market.db
EMBED_MODEL=BAAI/bge-small-en-v1.5
RAG_INDEX_DIR=data/faiss_index
DATA_DIR=data
LOG_LEVEL=INFO
```

---

## 8. Phased Milestones

| Phase | Scope | Deliverable |
|-------|-------|-------------|
| **0. Scaffold** | pyproject, package layout, config, CI lint | `pip install -e .` works |
| **1. Data store** | DDL, SQLAlchemy models, ingest CLI, query helpers, unit tests | `tradeagent ingest` populates SQLite |
| **2. Indicators + forecast** | indicators, signals, linear forecaster, feature materializer | `tradeagent features build`; tests pass |
| **3. Agent skeleton** | Deepseek client, tool registry with 3 tools, single-turn loop | `tradeagent ask` answers a price-history question |
| **4. Prompt chain** | planner → executor → synthesizer, run logging | multi-tool answers, traces saved |
| **5. RAG** | loader, embedder, FAISS index, retriever tool wired in | knowledge-grounded answers with citations |
| **6. Viz + report** | charts, markdown report builder | end-to-end report file per run |
| **7. Eval + polish** | walk-forward backtest of forecaster, agent eval set, README | reproducible demo |

Suggested order is sequential; phases 2 and 3 can be parallelized once phase 1 lands.

---

## 9. Testing Strategy

- **Unit**: indicators (known fixtures), DB queries (in-memory SQLite), tool input validation.
- **Integration**: end-to-end `agent.chain.run` against a mocked Deepseek client returning canned tool calls — verifies the chain wiring without burning API tokens.
- **Forecast eval**: walk-forward over last 2 years, log RMSE + directional accuracy per symbol to `forecasts` table; threshold gate for regressions.
- **Smoke**: a daily script that ingests, runs one agent query, and asserts a report file is produced.

---

## 10. Risks & Open Questions

1. **Deepseek API access** — confirm key + rate limits; have fallback (`openai` GPT-4o-mini) behind same client interface.
2. **Data source limits** — Alpha Vantage free tier is 25 req/day, 5/min; an initial backfill across the full NASDAQ universe will take days or require a premium tier. v1 scope should pin to a curated symbol list (e.g. top 50 NASDAQ tickers) and rely on incremental daily updates. Premium tier or a swap to Polygon/Tiingo is the upgrade path.
3. **Knowledge corpus copyright** — only ingest public-domain or permissively licensed sources; keep `knowledge_base/SOURCES.md`.
4. **Forecast over-claim** — surface confidence intervals and walk-forward R² in every answer; the agent prompt must disclaim "not financial advice."
5. **Schema scale** — long-format `features` table is flexible but can balloon; revisit wide-table or columnar (DuckDB/Parquet) if row count > 10M.
6. **Determinism** — set `temperature=0` for planner/synthesizer; allow slight temperature only in the final prose pass.

---

## 11. First Concrete Tasks (when implementation begins)

1. Create `pyproject.toml` with deps: `sqlalchemy`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `httpx` (Alpha Vantage client), `tenacity` (retry/backoff), `pydantic`, `pydantic-settings`, `typer`, `matplotlib`, `mplfinance`, `openai`, `sentence-transformers`, `faiss-cpu`, `pypdf`, `pytest`, `pytest-httpx`.
2. Write `tradeagent/config.py` + `.env.example` (including `ALPHAVANTAGE_*` keys).
3. Implement `data/db.py` (engine + `init_db()`), `data/models.py`, `data/schemas.sql`.
4. Implement `data/ingest.py`:
   - `AlphaVantageClient` (token-bucket rate limit, exponential backoff on `Note`/`Information` throttle, retries via `tenacity`),
   - `MarketDataClient` interface + CSV fallback,
   - incremental upsert into `ohlcv_bars`,
   - wire `tradeagent ingest` in `cli.py`.
5. Write `tests/test_ingest.py` (mock Alpha Vantage responses with `pytest-httpx`) and `tests/test_db.py` (insert + range query).

Everything after that follows the phase table above.
