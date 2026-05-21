from __future__ import annotations

import pytest

from tradeagent.agent.llm import make_llm_client


def test_default_provider_is_deepseek():
    client = make_llm_client()
    assert client.provider == "deepseek"
    assert client.model == "deepseek-chat"


def test_openai_provider_selected(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    from tradeagent import config

    config.get_settings.cache_clear()
    client = make_llm_client("openai")
    assert client.provider == "openai"
    assert client.model == "gpt-4o-mini"
    assert client.client.base_url.host == "api.openai.com"


def test_provider_from_env_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from tradeagent import config

    config.get_settings.cache_clear()
    client = make_llm_client()
    assert client.provider == "openai"


def test_model_override():
    client = make_llm_client("openai", model="gpt-4o")
    assert client.model == "gpt-4o"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        make_llm_client("anthropic")
