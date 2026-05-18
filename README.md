# TradingAgent

A reasoning trading-research agent over a SQL market-data store, with Alpha Vantage ingest, linear forecasting, signal decomposition, and a financial-knowledge RAG. The LLM brain is Deepseek-V3 via its OpenAI-compatible API.

See `docs/PLAN.md` for the full architecture and rationale.

## Quickstart

```powershell
# 1. Install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# 2. Configure
Copy-Item .env.example .env
# Edit .env: set DEEPSEEK_API_KEY and ALPHAVANTAGE_API_KEY

# 3. Ingest NASDAQ price history (free tier: 5 req/min, 25/day — pin to a small list)
tradeagent ingest AAPL MSFT NVDA --interval 1d

# 4. Materialize indicator features
tradeagent features build

# 5. Index the financial knowledge base for RAG
#    (drop PDFs / .md / .html under data/knowledge_base/ first)
tradeagent rag index data/knowledge_base

# 6. Ask
tradeagent ask "Is NVDA overbought right now, and what's a sensible 5-day forecast?"

# 7. Rebuild a markdown report from a prior run
tradeagent report 1
```

## Layout

```
tradeagent/
├── config.py              # pydantic-settings, reads .env
├── cli.py                 # typer entrypoint (tradeagent ...)
├── data/                  # SQL store: models, db, ingest, queries, features
├── forecast/              # indicators, FFT signals, ridge forecaster
├── agent/                 # Deepseek client, tool registry, planner→executor→synthesizer chain
├── rag/                   # loader, BGE embedder, FAISS vector store, retriever
└── viz/                   # matplotlib charts + markdown report builder
```

## Testing

```powershell
pip install -e ".[dev]"
pytest
```

The test suite uses an isolated temp SQLite per test (see `tests/conftest.py`) and mocks Alpha Vantage / Deepseek so it runs offline.

## Notes

- **Alpha Vantage free tier**: 25 req/day, 5/min. The ingest client rate-limits and backs off on `Note`/`Information` throttle responses. For broader universes, get a premium key or swap in a Polygon/Tiingo `MarketDataClient`.
- **No financial advice**: the agent appends a disclaimer; forecasts surface walk-forward R² as a confidence anchor — treat them as research signals, not recommendations.
