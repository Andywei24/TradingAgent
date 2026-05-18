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

    from tradeagent import config
    from tradeagent.data import db as db_mod

    config.get_settings.cache_clear()
    db_mod.get_engine.cache_clear()

    from tradeagent.data.db import init_db

    init_db()
    yield
    config.get_settings.cache_clear()
    db_mod.get_engine.cache_clear()
