from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point every test at a fresh temp dir + sqlite DB; reset settings cache."""
    db = tmp_path / "market.db"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "faiss_index"))
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    monkeypatch.setenv("ALPHAVANTAGE_RATE_LIMIT_PER_MIN", "60")
    monkeypatch.setenv("ALPHAVANTAGE_PREMIUM", "false")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("TAVILY_NEWS_DAYS", "7")
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5")
    # Pin LLM settings so tests don't depend on the developer's real .env.
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    from tradeagent import config
    from tradeagent.data import db as db_mod

    config.get_settings.cache_clear()
    db_mod.get_engine.cache_clear()

    from tradeagent.data.db import init_db

    init_db()
    yield
    config.get_settings.cache_clear()
    db_mod.get_engine.cache_clear()
