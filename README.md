# TradingAgent

A reasoning trading-research agent over a SQL market-data store, with Alpha Vantage ingest, linear forecasting, signal decomposition, and a financial-knowledge RAG. The LLM brain is pluggable — **Deepseek-V3 or OpenAI** out of the box, both via the OpenAI-compatible chat-completions protocol.

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
# Edit .env: set ALPHAVANTAGE_API_KEY, pick LLM_PROVIDER (deepseek|openai),
# set the matching key (DEEPSEEK_API_KEY or OPENAI_API_KEY), and optionally
# set TAVILY_API_KEY for recent market/news search.

# 3. Ingest NASDAQ price history (free tier: 5 req/min, 25/day — pin to a small list)
tradeagent ingest AAPL MSFT NVDA --interval 1d

# 4. Materialize indicator features
tradeagent features build

# 5. Index the financial knowledge base for RAG
#    (drop PDFs / .md / .html under data/knowledge_base/ first)
tradeagent rag index data/knowledge_base

# 6. Ask (uses LLM_PROVIDER from .env; override per-call with --provider)
# Analysis questions automatically include recent Tavily news when TAVILY_API_KEY is set.
tradeagent ask "Is NVDA overbought right now, and what's a sensible 5-day forecast?"
tradeagent ask "Compare AAPL vs MSFT momentum" --provider openai

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

## LLM providers

The agent talks to any OpenAI-compatible chat-completions endpoint. Two providers ship configured:

| Provider | Env key | Default model | Base URL |
|---|---|---|---|
| `deepseek` (default) | `DEEPSEEK_API_KEY` | `deepseek-chat` | `https://api.deepseek.com` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | `https://api.openai.com/v1` |

Set the default with `LLM_PROVIDER` in `.env`, or override per call with `tradeagent ask "..." --provider openai`. Programmatically: `run_chain(query, provider="openai")` or `make_llm_client("openai", model="gpt-4o")`. Add a third provider by extending `_provider_config` in `tradeagent/agent/llm.py`.

## News search

Set `TAVILY_API_KEY` to let the agent search recent market/news results during analysis-style questions. Defaults are `TAVILY_NEWS_DAYS=7` and `TAVILY_MAX_RESULTS=5`. If Tavily is not configured or unavailable, the agent continues with local SQL, indicators, forecasts, and RAG evidence instead of failing the run.

## Notes

- **Alpha Vantage free tier**: 25 req/day, 5/min. Several features are premium-gated — adjusted closes (`TIME_SERIES_DAILY_ADJUSTED`) and `outputsize=full`. The client defaults to the free `TIME_SERIES_DAILY` + `outputsize=compact` (latest ~100 daily bars; `adj_close == close`) and **fails fast** on a premium message instead of retrying it as a throttle. With a premium key, set `ALPHAVANTAGE_PREMIUM=true` to unlock adjusted closes and full history. For broader universes, get a premium key or swap in a Polygon/Tiingo `MarketDataClient`.
  - 100 bars is enough to run the forecaster, but walk-forward R² will often be low/negative on so short a window — that's an honest "low confidence" signal, not a bug. Premium full history sharpens it.
- **No financial advice**: the agent appends a disclaimer; forecasts surface walk-forward R² as a confidence anchor — treat them as research signals, not recommendations.
